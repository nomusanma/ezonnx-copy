from typing import List
import numpy as np
import matplotlib.pyplot as plt
def visualize_images(titles:List[str], images:List[np.ndarray]):
    """Visualize multiple images in a single row.

    Args:
        titles (List[str]): List of titles for each image.
        images (List[np.ndarray]): List of images to display.
    """
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 1:
        axes = [axes]
    for ax, title, image in zip(axes, titles, images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    plt.show()