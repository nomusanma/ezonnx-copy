import cv2
import numpy as np
from typing import Union, List, Optional
from ezonnx.core.inferencer import Inferencer
from ...data_classes.image_processing import ImageProcessingResult
from ...ops.preprocess import standard_preprocess, image_from_path

class RMBG14(Inferencer):
    """RMBG1.4 ONNX model for background removal.
    
    Args:
        quantize (Optional[str]): Quantization type, e.g., "quantized","fp16". Default is None.
        size (int): Input image size for the model. Default is 1024.

    Examples:
        ::

            from ezonnx import RMBG
            rmbg = RMBG(size=1024)
            result = rmbg("image.jpg")
            print(result.original_image)  # (H, W, 3)
            print(result.processed_image)  # (H, W, 3)    
    """

    def __init__(self, 
                 quantize:Optional[str]=None,
                 size:int=1024
                 )-> None:
        
        self._check_quantize(quantize, 
                             [None, "quantized","fp16"])

        repo_id = f"briaai/RMBG-1.4"
        filename = "onnx/model.onnx"
        self.sess = self._download_and_compile(repo_id, filename, quantize)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size

    def __call__(self,image:Union[str, np.ndarray])-> ImageProcessingResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.

        Returns:
            ImageProcessingResult: Inference result containing original and processed images.
        """
        image = image_from_path(image)

        tensor = self._preprocess(image)
        outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: tensor})
        mask = self._postprocess(outputs)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
        processed_image = (image * mask[..., None]).astype(np.uint8)

        return ImageProcessingResult(
            original_img=image,
            mask=mask,
            processed_img=processed_image
        )

    def _preprocess(self, image:np.ndarray):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array. 
        
        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """

        return standard_preprocess(image, 
                                   (self.size, self.size),
                                    std=(1, 1, 1),
                                    mean=(0.5, 0.5, 0.5))
    
    def _postprocess(self, outputs:List[np.ndarray]) -> np.ndarray:
        """Postprocess the model outputs.

        Args:
            outputs (List[np.ndarray]): Model outputs.
        
        Returns:
            np.ndarray: Postprocessed image array.(H, W)
        """
        mask = outputs[0][0][0]
        return mask
