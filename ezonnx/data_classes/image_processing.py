from .result import Result
import numpy as np  

class ImageProcessingResult(Result):
    """Image processing result containing original image and processed images.
    Background removal, super-resolution, low-light enhancement, etc.
    
    Attributes:
        original_img (np.ndarray): Original input image in BGR format.
        processed_img (np.ndarray): Processed image in BGR format.
        mask (np.ndarray | None): Optional mask.
        map (np.ndarray | None): Optional map.
    """
    
    processed_img: np.ndarray
    mask: np.ndarray | None = None  # Optional mask for inpainting tasks
    map: np.ndarray | None = None  # Optional map for tasks like depth estimation

    def _visualize(self) -> np.ndarray:
        """Return the processed image."""
        return self.processed_img