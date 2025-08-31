from typing import List,Optional,Union
import numpy as np
import matplotlib.pyplot as plt
def visualize_images(titles:Union[List[str],str], 
                     images:Union[List[np.ndarray],np.ndarray]):
    """Visualize multiple images in a single row.

    Args:
        titles (Union[List[str],str]): List of titles or title for each image.
        images (Union[List[np.ndarray],np.ndarray]): List of BGR images or image to display.
    """
    if isinstance(titles, str):
        titles = [titles]
    if isinstance(images, np.ndarray):
        images = [images]
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1:
        axes = [axes]
    
    # Ensure we have the same number of titles as images
    if len(titles) < len(images):
        titles = titles + [''] * (len(images) - len(titles))
    for ax, title, image in zip(axes, titles, images):
        if len(image.shape) == 2:
            ax.imshow(image, cmap='plasma')
        else:
            ax.imshow(image[..., ::-1])  # Convert BGR to RGB for displaying
        ax.set_title(title)
        ax.axis('off')
    plt.show()