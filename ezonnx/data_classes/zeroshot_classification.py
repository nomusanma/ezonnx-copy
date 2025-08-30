from typing import Optional,List
import numpy as np
from .result import Result
from ..ops.pca import create_pca_img


class ZeroshotClassificationResult(Result):
    """Zero-shot classification result containing image and text tokens.

    Attributes:
        original_img (np.ndarray): Original input image.
        size (int): Size of the input image.
        patch (int): Patch size used in the model.
        patch_tokens (np.ndarray): Feature tokens extracted from the image.
        class_token (np.ndarray): Class token extracted from the image.
        pca_img_rgb (np.ndarray): PCA applied image in RGB format.
        pca_img_gray (np.ndarray): PCA applied image in grayscale format.
        visualized_img (np.ndarray): Visualized original image.
    """
    size: Optional[int] = None
    patch: Optional[int] = None
    patch_tokens: Optional[np.ndarray] = None
    class_token: Optional[np.ndarray] = None
    prob: np.ndarray
    texts: List[str]
    text_tokens: Optional[np.ndarray] = None
    text_features: Optional[np.ndarray] = None

    def _visualize(self) -> np.ndarray:
        """visualize result to the original image."""
        return self.pca_img_rgb

    @property
    def pred_text(self) -> str:
        """Get the predicted text based on the highest probability."""
        if self.prob is None:
            raise ValueError("prob must be set before calling this method.")
        max_index = np.argmax(self.prob)
        return self.texts[max_index]
    
    @property
    def pred_idx(self) -> int:
        """Get the index of the predicted text based on the highest probability."""
        if self.prob is None:
            raise ValueError("prob must be set before calling this method.")
        return int(np.argmax(self.prob))
    
    @property
    def pca_img_rgb(self) -> np.ndarray:
        """get PCA applied image in RGB"""
        n_grid = self.size//self.patch
        return create_pca_img(n_grid, self.patch_tokens, n_components=3)
    
    @property
    def pca_img_gray(self) -> np.ndarray:
        """get PCA applied image in grayscale"""
        n_grid = self.size//self.patch
        return create_pca_img(n_grid, self.patch_tokens, n_components=1)