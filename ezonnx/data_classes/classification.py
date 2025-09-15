
import numpy as np
from ezonnx.data_classes.result import Result

class ClassificationResult(Result):
    """Image feature extraction result containing image and patch tokens.
    
    Attributes:
        original_img (np.ndarray): Original input image.
        class_id (np.ndarray): Predicted class index.
        score (np.ndarray): Predicted class score.
        logits (np.ndarray): Predicted class logits.
    """
    class_id: int
    score: float
    logits: np.ndarray

    def _visualize(self) -> None:
        """No visualization for classification result.
        
        Returns:
            None
        """
        return None


