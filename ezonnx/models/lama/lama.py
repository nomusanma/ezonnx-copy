from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ...data_classes.image_processing import ImageProcessingResult
from ...ops.preprocess import standard_preprocess, image_from_path, resize_with_aspect_ratio
from ...ops.postprocess import nms

class LaMa(Inferencer):
    """LaMa ONNX model for image inpaiting.

    Args:
        identifier (str): Model identifier, e.g., "s", "m", "l", "x".
        quantize (Optional[str]): Quantization type, e.g., "q4", "quantized", "fp16". Default is None.
        thresh (float): Confidence threshold for filtering detections. Default is 0.3.
        size (int): Input image size for the model. Default is 640. must be a multiple of 16.
    """

    def __init__(self,
                 onnx_path:Optional[str]=None):
        
        if onnx_path is None:
            # Initialize model
            repo_id = f"Carve/LaMa-ONNX"
            filename = f"lama_fp32.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_name_mask = self.sess.get_inputs()[1].name
        self.size = 512
    
    def __call__(self,
                 image:Union[str, np.ndarray],
                 mask:Union[str, np.ndarray]
                 )-> ImageProcessingResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
            mask (Union[str, np.ndarray]): Input mask path or mask array.
        
        Returns:
            ImageProcessingResult: Inference result containing boxes and classes.
        """
        image = image_from_path(image)
        self.input_wh = image.shape[:2][::-1]  # (w, h)
        mask = image_from_path(mask,gray_scale=True)
        target_size = (self.size, self.size)
        input_tensor = self._preprocess(image)
        mask_tensor = self._preprocess_mask(mask)
        outputs = self.sess.run(None,
                            {self.input_name: input_tensor,
                             self.input_name_mask: mask_tensor})
        processed_image = self._postprocess(outputs)

        return ImageProcessingResult(
            original_img=image,
            processed_img=processed_image,
            mask=mask
        )
    
    def _preprocess(self,image:np.ndarray
                    )-> np.ndarray:
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """
        return standard_preprocess(image, 
                                   (self.size, self.size),
                                   standardize=False)
    
    def _preprocess_mask(self,mask:np.ndarray
                    )-> Tuple[np.ndarray,float]:
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask_tensor = (mask[np.newaxis, ...]/255.0)[None]
        mask_tensor = ((mask_tensor > 0) * 1).astype("float32")
        return mask_tensor
    
    def _postprocess(self,
                     outputs:np.ndarray
                     )-> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Postprocess the model outputs to extract class labels and bounding boxes.

        Args:
            outputs (np.ndarray): outputs from the model containing bounding boxes and class scores.

        Returns:
            np.ndarray: Processed image in HWC format.
        """
        image = np.transpose(outputs[0][0],(1,2,0)).astype("uint8")
        image = cv2.resize(image, (self.input_wh), interpolation=cv2.INTER_LINEAR)
        return image
