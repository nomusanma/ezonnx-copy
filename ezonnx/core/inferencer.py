from typing import Union, List, Tuple, Any, Optional
import onnxruntime as ort
from pydantic import BaseModel
from abc import ABC, abstractmethod
from ezonnx.core.downloader import get_weights

class Inferencer(ABC):
    """DINOv3 ONNX model for image feature extraction.
    Args:
        model_path (str): Path to the ONNX model file.
        size (int): Input image size for the model. Default is 384. must be a multiple of 16.
        patch (int): Patch size used in the model. Default is 16.

    Examples:
        ::

            from dinov3 import DinoV3
            dino = DinoV3("weights/v3_s/model.onnx", size=384*4)
            result = dino("image.jpg")
            print(result.patch_token.shape)  # (N, D) D depends on the model
            print(result.class_token.shape)  # (1, 1)
            print(result.pca_image_rgb.shape)  # (H, W, 3)
            print(result.pca_image_gray.shape)  # (H, W)        
    """
    def _check_backbone(self, backbone: str, backbone_list:List[str]) -> None:
        if backbone not in backbone_list:
            raise ValueError(f"Invalid backbone type: {backbone}. Must be one of {backbone_list}.")

    def _check_quantize(self, quantize: Optional[str],quantize_list:List[Optional[str]] ) -> None:
        if quantize not in quantize_list:
            raise ValueError(f"Invalid quantization type: {quantize}. Must be one of {quantize_list}.")

    def _compile_from_path(self, model_path: str) -> ort.InferenceSession:
        return ort.InferenceSession(model_path, 
                                    providers=['CUDAExecutionProvider',
                                               'CPUExecutionProvider']
                                    )

    def _download_and_compile(self, 
                              repo_id: str, 
                              filename: str,
                              quantize: Optional[str] = None,
                              data:bool=False
                              ) -> ort.InferenceSession:
        
        if quantize is not None:
            filename = filename.replace(".onnx",f"_{quantize}.onnx")
        model_path = get_weights(repo_id, filename)

        if data:
            data_filename = filename+"_data"
            _ = get_weights(repo_id, data_filename)
        
        return ort.InferenceSession(model_path, 
                                    providers=['CUDAExecutionProvider',
                                               'CPUExecutionProvider']
                                    )

    @abstractmethod
    def __call__(self,*args,**kwargs)->BaseModel:
        pass

    @abstractmethod
    def _preprocess(self,*args,**kwargs)-> Any:
        pass

    @abstractmethod
    def _postprocess(self,*args,**kwargs)-> Any:
        pass
