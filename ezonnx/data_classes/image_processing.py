from pydantic import BaseModel, ConfigDict
import numpy as np  

class ImageProcessingResult(BaseModel):
    """Image processing result containing original image and processed images.
    Background removal, super-resolution, low-light enhancement, etc.
    
    Attributes:
        original_image (np.ndarray): Original input image in BGR format.
        processed_image (np.ndarray): Processed image in BGR format.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    original_img: np.ndarray
    processed_img: np.ndarray
    mask: np.ndarray | None = None  # Optional mask for inpainting tasks
    map: np.ndarray | None = None  # Optional map for tasks like depth estimation