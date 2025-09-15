import cv2
import numpy as np
from typing import Union, List, Tuple, Optional
from ...core.inferencer import Inferencer
from ...data_classes.image_processing import ImageProcessingResult
from ...ops.preprocess import standard_preprocess, image_from_path,resize_with_aspect_ratio

class ORMBG(Inferencer):
    """ORMBG ONNX model for person-background removal.
    
    Args:
        quantize (Optional[str]): Quantization type, e.g., "q4", "quantized". Default is None.
        onnx_path (Optional[str]): Path to the ONNX model file. If None, the model will be downloaded. Default is None.

    Examples:
        ::

            from ezonnx import ORMBG
            model = ORMBG()
            result = model("image.jpg")    
            print(result.processed_img)  # bg removed image (H, W, 3)
            print(result.mask)  # mask (H, W)  0 to 1  
    """

    def __init__(self, 
                 quantize:Optional[str]=None,
                 onnx_path:Optional[str]=None,
                 size:int=1024
                 )-> None:
        if onnx_path is None:
            self._check_quantize(quantize, 
                                [None, "q4", "quantized","fp16"])

            repo_id = f"onnx-community/ormbg-ONNX"
            filename = "onnx/model.onnx"
            self.sess = self._download_and_compile(repo_id, filename, quantize)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size

    def __call__(self,image:Union[str, np.ndarray])-> ImageProcessingResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.

        Returns:
            FeatureExtractionResult: Inference result containing class and patch tokens.
        """
        image = image_from_path(image)
        padded_image, ratio= resize_with_aspect_ratio(image,self.size)
        tensor = self._preprocess(padded_image)
        outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: tensor})
        mask = self._postprocess(outputs)
        mask = mask[:int(image.shape[0]*ratio), :int(image.shape[1]*ratio)]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
        processed_image = (image * mask[..., None]).astype(np.uint8)
        return ImageProcessingResult(
            original_img=image,
            processed_img=processed_image,
            mask=mask
        )

    def _preprocess(self, image:np.ndarray):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array. 
        
        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """
        
        return standard_preprocess(image, standardize=False)
    
    def _postprocess(self, outputs:List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess the model outputs to extract patch tokens and class tokens.

        Args:
            outputs (List[np.ndarray]): Model outputs containing patch tokens and class token.
        
        Returns:
            np.ndarray: Mask array. (H, W) with values 0 to 1.
        """
        mask = outputs[0][0][0]
        return mask
