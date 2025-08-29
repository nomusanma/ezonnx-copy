from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict
import numpy as np
from ..ops.postprocess import draw_masks

class SAMSegmentationResult(BaseModel):
    """Data class for segmentation results.

    Args:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        masks (Dict[int,np.ndarray]): Dictionary of binary masks with label ids as keys.
        boxes (Optional[List[np.ndarray]]): List of bounding boxes for each segmented object in shape (4,). Default is None.
        labels (Optional[List[int]]): List of label ids corresponding to each mask. Default is None.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    original_img: np.ndarray
    masks: Dict[int,np.ndarray]
    boxes: Optional[List[np.ndarray]] = None

    @property
    def visualized_img(self) -> np.ndarray:
        """Get the processed image with segmentation masks applied.

        Returns:
            np.ndarray: Processed image in shape (H, W, 3). BGR
        """
        return draw_masks(self.original_img,
                        self.masks,
                        draw_border=False)
