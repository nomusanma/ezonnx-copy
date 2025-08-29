from sklearn.decomposition import PCA
import numpy as np

def create_pca_img(n_grid,patch_tokens,n_components=3) -> np.ndarray:
    """convert patch_token to PCA image.
    
    Args:
        n_grid (int): Number of patches in one dimension (H or W).
        patch_tokens (np.ndarray): Patch tokens extracted from the image.(N, D)
        n_components (int): 1 or 3. If 1, returns grayscale image.
    
    Returns:
        np.ndarray: PCA image in shape (H, W, 3) or (H, W) depending on n_components.
    """
    # check length of patch_tokens is equal to n_grid*n_grid
    if patch_tokens.shape[0] != n_grid * n_grid:
        raise ValueError(f"Length of patch_tokens {patch_tokens.shape[0]} does not match n_grid*n_grid {n_grid * n_grid}.")
    # dimension reduction by PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(patch_tokens) # (N, 3)

    # (N, 3) to (H,W,3)
    if n_components ==3:
        image = features_pca.reshape(n_grid, n_grid, 3)
    elif n_components == 1:
        image = features_pca.reshape(n_grid, n_grid)
    else:
        raise ValueError("n_components must be 1 or 3")

    # create an image
    image_min, image_max = image.min(), image.max()
    image_norm = (image - image_min) / (image_max - image_min)*255.0
    image_uint8 = np.clip(image_norm, 0, 255).astype(np.uint8)
    return image_uint8