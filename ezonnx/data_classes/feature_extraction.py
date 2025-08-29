from typing import Optional
from pydantic import BaseModel, ConfigDict
from ..ops.pca import create_pca_img
import cv2
import numpy as np

class FeatureExtractionResult(BaseModel):
    """Image feature extraction result containing image and patch tokens.
    
    Attributes:
        original_img (np.ndarray): Original input image.
        size (int): Size of the input image.
        patch (int): Patch size used in the model.
        patch_tokens (np.ndarray): Feature tokens extracted from the image.
        class_token (np.ndarray): Class token extracted from the image.
        pca_image_rgb (np.ndarray): PCA applied image in RGB format.
        pca_image_gray (np.ndarray): PCA applied image in grayscale format.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    original_img: np.ndarray
    size: int
    patch: int
    patch_tokens: np.ndarray
    class_token: Optional[np.ndarray] = None

    @property
    def visualized_img(self) -> np.ndarray:
        """Get the PCA image in RGB format.

        Returns:
            np.ndarray: PCA image in shape (H, W, 3). RGB
        """
        return self.pca_img_rgb
    
    @property
    def pca_img_rgb(self) -> np.ndarray:
        """get PCA applied image in RGB"""
        n_grid = self.size//self.patch
        img =  create_pca_img(n_grid, self.patch_tokens, n_components=3)
        return cv2.resize(img, self.original_img.shape[1::-1])
    @property
    def pca_img_gray(self) -> np.ndarray:
        """get PCA applied image in grayscale"""
        n_grid = self.size//self.patch
        img = create_pca_img(n_grid, self.patch_tokens, n_components=1)
        return cv2.resize(img, self.original_img.shape[1::-1])
