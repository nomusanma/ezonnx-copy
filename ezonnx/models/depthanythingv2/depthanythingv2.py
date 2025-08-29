import cv2
import numpy as np
from typing import Union, List, Tuple, Optional
from ezonnx.core.inferencer import Inferencer
from ...data_classes.image_processing import ImageProcessingResult
from ...ops.preprocess import standard_preprocess, image_from_path

class DepthAnythingV2(Inferencer):
    """DepthAnythingV2 ONNX model for monocular depth estimation.
    
    Args:
        backbone (str): model backbone type, e.g., "vits16", "vits16plus", "vitb16", "vitl16".
        quantize (Optional[str]): Quantization type, e.g., "q4", "quantized". Default is None.
        size (int): Input image size for the model. Default is 384. must be a multiple of 16.
        patch (int): Patch size used in the model. Default is 16.

    Examples:
        ::

            from dinov3 import DinoV3
            dino = DinoV3("vits", size=1024)
            result = dino("image.jpg")
            print(result.patch_token.shape)  # (N, D) D depends on the model
            print(result.class_token.shape)  # (1, 1)
            print(result.pca_image_rgb.shape)  # (H, W, 3)
            print(result.pca_image_gray.shape)  # (H, W)        
    """

    def __init__(self, 
                 backbone:str, 
                 quantize:Optional[str]=None,
                 size:int=518
                 )-> None:
        
        self._check_backbone(backbone, 
                            ["vits16", "small","base","large"])
        self._check_quantize(quantize, 
                             [None, "q4", "quantized","fp16"])

        repo_id = f"onnx-community/depth-anything-v2-{backbone}"
        filename = "onnx/model.onnx"
        self.sess = self._download_and_compile(repo_id, filename, quantize)
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

        tensor = self._preprocess(image)
        outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: tensor})
        depth,depth_image = self._postprocess(outputs)
        processed_image = cv2.resize(depth_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        return ImageProcessingResult(
            original_img=image,
            processed_img=processed_image,
            map=depth
        )

    def _preprocess(self, image:np.ndarray):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array. 
        
        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """

        return standard_preprocess(image, (self.size, self.size))
    
    def _postprocess(self, outputs:List[np.ndarray]) -> np.ndarray:
        """Postprocess the model outputs to extract patch tokens and class tokens.

        Args:
            outputs (List[np.ndarray]): Model outputs containing patch tokens and class token.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: patch tokens and class token.
        """
        depth = outputs[0][0]
        # normalize to float 0~1
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_image = (depth * 255).astype("uint8")
        return depth, depth_image
