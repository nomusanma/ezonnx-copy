from typing import Tuple, Union
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ...ops.preprocess import image_from_path
from ...data_classes.keypoint_detection import KeypointDetectionResult
from ...core.inferencer import Inferencer

class LightGlue(Inferencer):
    """LightGlue ONNX model for local feature matching.

    Args:
        identifier (str): Model identifier, e.g., "aliked"
    """

    def __init__(self,identifier):
        self._check_backbone(identifier,["aliked"])
        # Initialize model
        repo_id = f"bukuroo/ALIKED-LightGlue-ONNX"
        filename = f"lightglue_for_{identifier}.onnx"
        self.sess = self._download_and_compile(repo_id, filename)

    def __call__(self,
                 kpts0,kpts1,
                 descs0,descs1,
                 m_thresh:float=0.2
                 )-> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Run inference on the input keypoints and descriptors.
        Number of keypoints in image 0 and 1 must be the same as those used in LightGlue model.

        Args:
            kpts0 (np.ndarray): Normalized(-1 to 1) Keypoints from image 0, shape (N, 2).
            kpts1 (np.ndarray): Normalized(-1 to 1) Keypoints from image 1, shape (N, 2).
            descs0 (np.ndarray): Descriptors from image 0, shape (N, D).
            descs1 (np.ndarray): Descriptors from image 1, shape (N, D).
            m_thresh (float): Matching confidence threshold.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Matched pairs and their confidence scores.
        """
        matches,confs = self.sess.run(None,
            {
                "kpts0": kpts0[None,:], # add batch dimension
                "kpts1": kpts1[None,:],
                "desc0": descs0[None,:],
                "desc1": descs1[None,:],
            },)
        mask = confs>m_thresh
        return matches[mask], confs[mask]
    
    def _preprocess(self):
        pass

    def _postprocess(self):
        pass