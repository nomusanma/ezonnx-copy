from typing import Tuple, Union
import cv2
import numpy as np

def image_from_path(image: Union[str, np.ndarray],
                    gray_scale=False) -> np.ndarray:
    """Load image from file path or return the image if it's already an array.

    Args:
        image (Union[str, np.ndarray]): Input image path or image array.
    
    Returns:
        np.ndarray: Loaded image array in BGR format.
    """
    if gray_scale:
        flag = cv2.IMREAD_GRAYSCALE
    else:
        flag = cv2.IMREAD_COLOR
    if isinstance(image, str):
        # get image from file path
        image = cv2.imread(image, flag)
    if image is None:
        raise ValueError("Failed to read image. Check the file path or image format.")
    return image

def standard_preprocess(image: np.ndarray, 
                        size: Tuple[int,int],
                        standardize: bool = True,
                        dim_order: str = 'CHW',
                        mean : Tuple[float,float,float] = (0.485, 0.456, 0.406),
                        std : Tuple[float,float,float] = (0.229, 0.224, 0.225)
                        ) -> np.ndarray:
    """Preprocess the input image for the model.

    Args:
        image (np.ndarray): Input image array. 
        size (Tuple[int]): Target size for the image (height, width).
        standardize (bool): Whether to standardize the image. Default is True.
        dim_order (str): Dimension order for the output tensor. 'CHW' for channel
    
    Returns:
        np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
    """

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    image = cv2.resize(image, size)
    # Convert to float32 and scale to [0,1]
    image = image.astype(np.float32) / 255.0
    if standardize:
        # Normalize: channel-wise normalization
        std = np.array(std, dtype=np.float32)
        mean = np.array(mean, dtype=np.float32)
        image = (image - mean) / std
    if dim_order == 'CHW':
        # Convert HWC to CHW
        image = np.transpose(image, (2, 0, 1))
    # Add batch dimension to get (1, 3, H, W)
    image = np.expand_dims(image, axis=0)
    return image

def resize_with_aspect_ratio(image:np.ndarray, 
                             size:int, 
                             interpolation=cv2.INTER_LINEAR
                             ) -> Tuple[np.ndarray, float]:
    """Resizes an image while maintaining aspect ratio and pads it.
    
    Args:
        image (np.ndarray): Input image array.
        size (int): Desired size for the longest side of the image.
        interpolation: Interpolation method for resizing. Default is cv2.INTER_LINEAR.
    
    Returns:
        Tuple[np.ndarray, float]: Resized and padded image, and the scaling ratio.
    """
    original_height, original_width = image.shape[:2]
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    # Create a new image with the desired size and pad the resized image onto it
    padded_image = np.zeros((size, size, 3), dtype=np.uint8)  # Assuming 3 channels (RGB)
    # x_offset = (size - new_width) // 2
    # y_offset = (size - new_height) // 2
    padded_image[:new_height, : new_width] = resized_image

    return padded_image, ratio