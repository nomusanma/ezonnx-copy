from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from .pponnxcr import TextSystem

from ...core.downloader import get_weights
from ...data_classes.ocr import BoxedResult,OCRResult
from ...ops.preprocess import standard_preprocess, image_from_path

class PPOCR(Inferencer):
    """PPOCR ONNX model for optical character recognition.

    Args:
        identifier (str): Model identifier, e.g., "v5-server","v5-mobile".
        box_thresh (float): Confidence threshold for text detection. Default is 0.6.
        unclip_ratio (float): Unclip ratio for detection box. Default is 1.3.
        onnx_path (Optional[str]): Path to a local ONNX model file. If provided, the model will be loaded from this path instead of downloading. Default is None.
    
    Examples:
        Usage example:
        ::
            from ezonnx import PPOCR
            model = PPOCR("v5-server")  # you can choose "v5-server","v5-mobile"
            result = model("image.jpg")
            for box in result.boxed_results:
                print(box.box)  # (4,2) array of box coordinates
                print(box.text)  # recognized text
                print(box.score)  # confidence score
            print(result.visualized_img)  # (H, W, 3) image with boxes and text drawn
    """
    def __init__(self,
                 identifier:str="v5-server",
                 box_thresh=0.6, 
                 unclip_ratio=1.3,
                 onnx_path:Optional[str]=None):

        # build
        self._check_backbone(identifier,["v5-server","v5-mobile"])
        if onnx_path is None:
            # Initialize model
            repo_id = f"bukuroo/PPOCRv5-ONNX"
            filename_dict = f"ppocr{identifier.split('-')[0]}_dict.txt"
            filename_det = f"ppocr{identifier}-det.onnx"
            filename_rec = f"ppocr{identifier}-rec.onnx"
            filename_cls = f"ppocr{identifier.split('-')[0]}-cls.onnx"
            self.sess_det = self._download_and_compile(repo_id, filename_det)
            self.sess_rec = self._download_and_compile(repo_id, filename_rec)
            self.sess_cls = self._download_and_compile(repo_id, filename_cls)
            self.char_dict_path = get_weights(repo_id, filename_dict)
            self.text_system = TextSystem(
                self.sess_det, self.sess_rec, self.sess_cls,
                self.char_dict_path,
                use_angle_cls=True,
                box_thresh=box_thresh,
                unclip_ratio=unclip_ratio,
            )
        else:
            self.sess = self._compile_from_path(onnx_path)
        
    def __call__(self,image:Union[str, np.ndarray])-> OCRResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            List[Dict[str, Any]]: OCR results with bounding boxes and recognized text.
        """
        image = image_from_path(image)
        boxed_results = self.text_system.detect_and_ocr(image) # List[BoxedResult]

        return OCRResult(
            original_img=image,
            boxed_results=boxed_results) 
    
    def _preprocess(self)-> None:
        pass

    def _postprocess(self)-> None:
        pass