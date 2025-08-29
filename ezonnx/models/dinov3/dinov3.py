import cv2
import numpy as np
from typing import Union, List, Tuple, Optional
from ezonnx.core.inferencer import Inferencer
from ...data_classes.feature_extraction import FeatureExtractionResult
from ...ops.preprocess import standard_preprocess

class DinoV3(Inferencer):
    """DINOv3 ONNX model for image feature extraction.

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
                 size:int=384, 
                 patch:int=16
                 )-> None:
        if backbone not in ["vits16", "vits16plus", "vitb16", "vitl16"]:
            raise ValueError(f"Invalid backbone type: {backbone}. Must be one of ['vits16', 'vits16plus', 'vitb16', 'vitl16'].")
        if size % 16 != 0:
            raise ValueError(f"Input size must be a multiple of 16. Given size: {size}.")
        if quantize not in [None, "q4", "quantized"]:
            raise ValueError(f"Invalid quantization type: {quantize}. Must be one of [None, 'q4', 'q8'].")
        
        repo_id = f"onnx-community/dinov3-{backbone}-pretrain-lvd1689m-ONNX"
        filename = "onnx/model.onnx"
        self.sess = self._download_and_compile(repo_id, filename, quantize, data=True)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size
        self.patch = patch

    def __call__(self,image:Union[str, np.ndarray])-> FeatureExtractionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.

        Returns:
            FeatureExtractionResult: Inference result containing class and patch tokens.
        """

        return self._infer(image)

    def _infer(self, image:Union[str, np.ndarray]) -> FeatureExtractionResult:
        """Run inference on the input image.
        
        Args:
            image (Union[str, np.ndarray]): Input image path or image array.

        Returns:
            FeatureExtractionResult: Inference result containing class and patch tokens.
        """

        if isinstance(image, str):
            # get image from file path
            image = cv2.imread(image)
        if image is None:
            raise ValueError("Failed to read image. Check the file path or image format.")
        tensor = self._preprocess(image)
        outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: tensor})
        feature_map,cls_token = self._postprocess(outputs)

        return FeatureExtractionResult(
            size=self.size,
            patch=self.patch,
            original_img=image,
            patch_tokens=feature_map,
            class_token=cls_token
        )

    def _preprocess(self, image:np.ndarray):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array. 
        
        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """

        return standard_preprocess(image, (self.size, self.size))
    
    def _postprocess(self, outputs:List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess the model outputs to extract patch tokens and class tokens.

        Args:
            outputs (List[np.ndarray]): Model outputs containing patch tokens and class token.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: patch tokens and class token.
        """
        patch_tokens = outputs[0][0][5:] # Skip the first 5 tokens (class token and register tokens)
        class_token = outputs[1][0]
        return patch_tokens, class_token
