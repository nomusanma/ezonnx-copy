from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from ezonnx.core.inferencer import Inferencer
from ...data_classes.object_detection import ObjectDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path, resize_with_aspect_ratio


class DFINE(Inferencer):
    """D-FINE ONNX model for object detection.

    Args:
        identifier (str): Model identifier, e.g., "s", "m", "l", "x".
        quantize (Optional[str]): Quantization type, e.g., "q4", "quantized", "fp16". Default is None.
        threshold (float): Confidence threshold for filtering detections. Default is 0.3.
        size (int): Input image size for the model. Default is 640. must be a multiple of 32.
    """

    def __init__(self,
                 identifier,
                 quantize=None,
                 threshold:float=0.3,
                 size=640):
        self._check_backbone(identifier,["s","m","l","x"])
        self._check_quantize(quantize,
                             [None,"q4","quantized","fp16"])
        # Initialize model
        repo_id = f"onnx-community/dfine_{identifier}_obj365-ONNX"
        filename = f"onnx/model.onnx"
        self.sess = self._download_and_compile(repo_id, filename, quantize)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size
        self.threshold = threshold
    
    def __call__(self,image:Union[str, np.ndarray])-> ObjectDetectionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            ObjectDetectionResult: Inference result containing boxes and classes.
        """
        image = image_from_path(image)
        input_tensor,ratio = self._preprocess(image)
        logits, boxes = self.sess.run(None,
                            {self.input_name: input_tensor})
        boxes, classes, scores = self._postprocess(logits, boxes,ratio)

        return ObjectDetectionResult(
            original_img=image,
            boxes=boxes,
            classes=classes,
            scores=scores
        )
    
    def _preprocess(self,image:np.ndarray)-> Tuple[np.ndarray,float]:
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded_image, ratio= resize_with_aspect_ratio(image,self.size)
        tensor = padded_image/255.0
        tensor = np.transpose(tensor,[2,0,1])
        return tensor[None,:].astype("float32"),ratio
    
    def _postprocess(self,
                     logits:np.ndarray, 
                     boxes:np.ndarray,
                     ratio:float)-> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Postprocess the model outputs to extract class labels and bounding boxes.

        Args:
            logits (np.ndarray): Model output logits for class predictions.
            boxes (np.ndarray): Model output bounding boxes.

        Returns:
            classes (np.ndarray): Predicted class labels for each box.
            boxes (np.ndarray): Corresponding bounding boxes.
            scores (np.ndarray): Confidence scores for each box.
        """
        logits = logits[0]  # Remove batch dimension

        boxes = boxes[0]  # Remove batch dimension
        #convert xcycwh to xyxy
        boxes[:,0] = boxes[:,0] - boxes[:,2]/2
        boxes[:,1] = boxes[:,1] - boxes[:,3]/2
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]
        # scale boxes back to original image size
        boxes = boxes * self.size / ratio

        # Calculate scores using softmax
        scores = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        # Get the best class and its score for each box
        classes = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        # Filter out low-confidence detections
        mask = scores >= self.threshold
        boxes = boxes[mask]
        classes = classes[mask]
        scores = scores[mask]

        return boxes, classes, scores