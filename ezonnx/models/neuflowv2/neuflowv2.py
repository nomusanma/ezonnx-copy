from typing import Optional, Tuple, Union
import cv2
import numpy as np
from ...core.inferencer import Inferencer
from ...ops.preprocess import image_from_path,standard_preprocess
from ...data_classes.optical_flow import OpticalFlowResult

class NeuFlowV2(Inferencer):
    """NeuFlowV2 ONNX model for optical flow estimation.

    Args:
        identifier (str): Model identifier, e.g.,"mixed","sintel","things".
    """

    def __init__(self,
                 identifier:str="mixed",):
        self._check_backbone(identifier,["mixed","sintel","things"])
        # Initialize model
        repo_id = f"bukuroo/NeuFlowV2-ONNX"
        filename = f"neuflow_{identifier}.onnx"
        self.sess = self._download_and_compile(repo_id, filename)
        self._get_input_details()
        self._get_output_details()

    def __call__(self, 
                 img_prev:Union[str, np.ndarray], 
                 img_now:Union[str, np.ndarray]
                 ) -> OpticalFlowResult:
        img_prev = image_from_path(img_prev)
        img_now = image_from_path(img_now)

        input_tensors = self._preprocess(img_prev, img_now)
        outputs = self.sess.run(self.output_names, 
                                {self.input_names[0]: input_tensors[0],
                                self.input_names[1]: input_tensors[1]})

        flow = self._postprocess(outputs[0])
        return OpticalFlowResult(
            previous_img=img_prev,
            original_img=img_now,
            flow=flow
        )

    def _preprocess(self, img_prev: np.ndarray, img_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.img_height, self.img_width = img_now.shape[:2]

        input_prev = standard_preprocess(img_prev,standardize=False,
                            size=(self.input_width,self.input_height))#self._prepare_input(img_now)
        input_now = standard_preprocess(img_now,standardize=False,
                            size=(self.input_width,self.input_height))
        

        return input_prev, input_now

    def _prepare_input(self, img: np.ndarray) -> np.ndarray:
        # Resize input image
        input_img = cv2.resize(img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _postprocess(self, output) -> np.ndarray:
        flow = output.squeeze().transpose(1, 2, 0)

        return cv2.resize(flow, (self.img_width, self.img_height))

    def _get_input_details(self):
        model_inputs = self.sess.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

    def _get_output_details(self):
        model_outputs = self.sess.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
