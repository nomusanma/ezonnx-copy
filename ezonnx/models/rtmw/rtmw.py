from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ..rtmdet.rtmdet import RTMDet
from ..rtmpose.rtmpose import RTMPose
from ...data_classes.object_detection import ObjectDetectionResult,PoseDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path

class RTMW(RTMPose):
    def __init__(self,
                 identifier:str="l-384",
                 person_detector:Inferencer=None,
                 kpt_thresh = 0.3,
                 iou_thresh = 0.6,
                 onnx_path:Optional[str]=None):
        # person detector
        if person_detector is None:
            self._person_det = RTMDet("m-person",iou_thresh=iou_thresh)
        else:
            self._person_det = person_detector
        
        # build
        self._check_backbone(identifier,["l-384","m-256","l-256","x-384"])
        if onnx_path is None:
            # Initialize model
            repo_id = f"bukuroo/RTMW-ONNX"
            filename = f"rtmw-{identifier}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        h,w = self.sess.get_inputs()[0].shape[2:]
        self.input_size_wh = (w,h)
        self.kpt_thresh = kpt_thresh
