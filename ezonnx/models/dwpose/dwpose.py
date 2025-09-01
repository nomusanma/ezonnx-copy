from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ..rtmdet.rtmdet import RTMDet
from ..rtmpose.rtmpose import RTMPose
from ...data_classes.object_detection import ObjectDetectionResult,PoseDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path

class DWPose(RTMPose):
    def __init__(self,
                 identifier:str="ll",
                 person_detector:Inferencer=None,
                 kpt_thresh = 0.3,
                 onnx_path:Optional[str]=None):
        # person detector
        if person_detector is None:
            self._person_det = RTMDet("m-person")
        else:
            self._person_det = person_detector
        
        # build
        self._check_backbone(identifier,["ll"])
        if onnx_path is None:
            # Initialize model
            repo_id = f"bukuroo/DWPose-ONNX"
            filename = f"dwpose-{identifier}-wholebody.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        h,w = self.sess.get_inputs()[0].shape[2:]
        self.input_size_wh = (w,h)
        self.kpt_thresh = kpt_thresh
