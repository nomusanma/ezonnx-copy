from typing import List, Optional, Dict
import cv2
import numpy as np
from .result import Result

from ..ops.postprocess import draw_kpts

class ImageMatchingResult(Result):
    """Data class for segmentation results.

    Attributes:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        query_img (np.ndarray): Query input image in shape (H, W, 3). BGR
        m_kpts0 (np.ndarray): Matched keypoints in the original image. Shape (M, 2).
        m_kpts1 (np.ndarray): Matched keypoints in the query image. Shape (M, 2).
        scores (np.ndarray): Confidence scores for each match. Shape (M,).
        visualized_img (Optional[np.ndarray]): Visualized image with keypoints drawn. Shape (H, W, 3). BGR
    """
    query_img: np.ndarray # (H, W, 3) BGR
    m_kpts0: np.ndarray # (M, 2) matched keypoints in the original image
    m_kpts1: np.ndarray # (M, 2) matched keypoints in the query image
    scores: np.ndarray # (M,) confidence scores for each match

    def _visualize(self) -> np.ndarray:
        """Get the processed image with segmentation masks applied.

        Returns:
            np.ndarray: Processed image in shape (H, W, 3). BGR
        """
        return self._draw_matches(self.original_img,self.query_img,
                          self.m_kpts0,self.m_kpts1)
    
    def _draw_matches(self, img1, img2, kpts1, kpts2):
        '''Draw matches between two images.
        Args:
            img1: First image.
            img2: Second image.
            kpts1: Matched keypoints in the first image.
            kpts2: Matched keypoints in the second image.

        Returns:
            out_img: Output image with matches drawn.
        '''
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
        out_img[:h1, :w1] = img1
        out_img[:h2, w1:w1 + w2] = img2

        for (x1, y1), (x2, y2) in zip(kpts1, kpts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(out_img, (int(x1), int(y1)), 4, color, -1)
            cv2.circle(out_img, (int(x2) + w1, int(y2)), 4, color, -1)
            cv2.line(out_img, (int(x1), int(y1)), (int(x2) + w1, int(y2)), color, 1)
        return out_img