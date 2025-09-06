from typing import List, Optional, Dict
import cv2
import numpy as np
from .result import Result

from ..ops.postprocess import draw_kpts

class KeypointDetectionResult(Result):
    """Data class for segmentation results.

    Attributes:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        kpts (np.ndarray): List of keypoints in shape (N, 2). Default is None.
        kpts_norm (np.ndarray): List of normalized keypoints in shape (N, 2). Default is None.
        scores (np.ndarray): List of confidence scores corresponding to each keypoint. Default is None.
        descs (np.ndarray): List of descriptors to each keypoint in shape(N, D). Default is None.
        visualized_img (np.ndarray): Processed image with all keypoints drawn.
    """
    kpts: np.ndarray # (N, 2) cordinates of keypoints
    kpts_norm: np.ndarray # (N, 2) normalized cordinates of keypoints (-1 to 1) to model input size
    scores: np.ndarray # (N, ) confidence scores of keypoints
    descs: np.ndarray # (N, D) descriptors of keypoints

    def _visualize(self) -> np.ndarray:
        """Get the processed image with segmentation masks applied.

        Returns:
            np.ndarray: Processed image in shape (H, W, 3). BGR
        """
        return draw_kpts(self.original_img,
                          self.kpts,
                          self.scores)