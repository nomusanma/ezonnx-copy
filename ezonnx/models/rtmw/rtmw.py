from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ..rtmdet.rtmdet import RTMDet
from ..rtmpose.rtmpose import RTMPose
from ...data_classes.object_detection import ObjectDetectionResult,PoseDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path

class RTMW(RTMPose):
    """RTMW ONNX model for whole body estimation.
    For person detection, RTMDet "m-person" model will be used by default.

    Args:
        identifier (str): Model identifier, e.g., "l-384","m-256","l-256","x-384"
        person_detector (Optional[Inferencer]): Pre-trained person detector. If None, RTMDet "m-person" model will be used.
        kpt_thresh (float): Keypoint confidence threshold for filtering keypoints. Default is 0.3.
        iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS) in person detection. Default is 0.6.
        onnx_path (Optional[str]): Path to a local ONNX model file. If provided, the model will be loaded from this path instead of downloading. Default is None.
    
    Examples:
        Usage example:
        ::
            from ezonnx import RTMW
            model = RTMW("l-384")  # you can choose "l-384","m-256","l-256","x-384"
            result = model("image.jpg")
            print(result.boxes)  # (N, 4) array of bounding boxes
            print(result.kpts)  # (N,17,2) array of keypoints for each box
            print(result.scores)  # (N,) array of confidence scores
            print(result.kpt_scores)  # (N,17) array of keypoint confidence scores
            print(result.visualized_img)  # (H, W, 3) image with keypoints and skeleton drawn

    """
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
