from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ..rtmdet.rtmdet import RTMDet
from ...data_classes.image_processing import ImageProcessingResult
from ...ops.preprocess import standard_preprocess, image_from_path

class RealESRGAN(Inferencer):
    """Real-ESRGAN ONNX model for image super-resolution.

    Args:
        identifier (str): Model identifier, e.g., "x4plus"".
        size (int): Input image size for the model. Default is 512. must be a multiple of 4.
    """

    def __init__(self,
                 identifier="x4plus",
                 onnx_path:Optional[str]=None,
                 size=128,
                 ):
        
        quantize=None
        # Initialize model
        if onnx_path is None:
            self._check_backbone(identifier,["x4plus"])
            repo_id = f"bukuroo/RealESRGAN-ONNX"
            filename = f"real-esrgan-{identifier}-{size}.onnx"
            self.sess = self._download_and_compile(repo_id, filename,quantize)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size
    
    def __call__(self,image:Union[str, np.ndarray])-> np.ndarray:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            np.ndarray: Super-resolved image array.
        """
        image = image_from_path(image)
        input_tensor = self._preprocess(image)
        output=self.sess.run(None,
                            {self.input_name:input_tensor})
        output_img = self._postprocess(output)
        return ImageProcessingResult(
            original_img=image,
            processed_img=output_img
        )
    
    def _preprocess(self,image:np.ndarray)-> np.ndarray:
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array.
        
        Returns:
            np.ndarray: Preprocessed image tensor.
        """
        input_tensor = standard_preprocess(image,standardize=False,
                            size=(self.size,self.size))
        return input_tensor

    def _postprocess(self,output:List[np.ndarray])-> np.ndarray:
        """Postprocess the model output to obtain the super-resolved image.

        Args:
            output (List[np.ndarray]): Raw output from the model.
        
        Returns:
            np.ndarray: Super-resolved image array. (H, W, 3) in RGB format.
        """
        hr_image = output[0]
        hr_image = np.squeeze(hr_image)
        hr_image = np.clip((hr_image * 255), 0, 255).astype(np.uint8)
        hr_image = hr_image.transpose(1, 2, 0)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR)
        return hr_image