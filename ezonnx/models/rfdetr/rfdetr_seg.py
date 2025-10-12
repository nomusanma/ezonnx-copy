from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ezonnx.data_classes.object_detection import InstanceSegmentationResult
from ezonnx.ops.preprocess import standard_preprocess, image_from_path, resize_with_aspect_ratio
from ezonnx.ops.postprocess import sigmoid,xywh2xyxy

class RFDETRSeg(Inferencer):
    """RF-DETR ONNX model for object detection.

    Args:
        identifier (str): Model identifier, e.g., "preview".
        thresh (float): Confidence threshold for filtering detections. Default is 0.3.
        onnx_path (Optional[str]): Path to a local ONNX model file. If provided, the model will be loaded from this path instead of downloading. Default is None.
    
    Examples:
        Usage example:
        ::

            from ezonnx import RFDETRSeg

            rfdetr = RFDETRSeg("preview")  # you can choose "preview"
            result = rfdetr("image.jpg")

            print(result.boxes)  # (N, 4) array of bounding boxes
            print(result.classes)  # (N,) array of class ids
            print(result.scores)  # (N,) array of confidence scores
            print(result.masks)  # (N, H, W) array of segmentation masks
            print(result.visualized_img)  # (H, W, 3) image with
    
    """

    def __init__(self,
                 identifier:Optional[str]=None,
                 thresh:float=0.3,
                 onnx_path:Optional[str]=None):
        
        if onnx_path is None and identifier is not None:
            self._check_backbone(identifier,["preview"])
            # Initialize model
            repo_id = f"bukuroo/RF-DETR-Seg-ONNX"
            filename = f"rf-detr-seg-{identifier}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = self.sess.get_inputs()[0].shape[2]
        self.thresh = thresh

    def __call__(self,image:Union[str, np.ndarray])-> InstanceSegmentationResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.

        Returns:
            InstanceSegmentationResult: Inference result containing boxes, classes, scores, and masks.
        """
        image = image_from_path(image)

        input_tensor = self._preprocess(image)
        outputs = self.sess.run(None,
                            {self.input_name: input_tensor})
        boxes, classes, scores, masks = self._postprocess(outputs,image.shape)

        return InstanceSegmentationResult(
            original_img=image,
            boxes=boxes,
            classes=classes,
            scores=scores,
            masks=masks
        )
    
    def _preprocess(self,image:np.ndarray
                    )-> np.ndarray:
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """
        padded_image, _ = resize_with_aspect_ratio(image,self.size)
        input_tensor = standard_preprocess(padded_image)
        return input_tensor
    
    def _postprocess(self,
                     outputs:np.ndarray,
                    original_shape:Tuple[int,int]
                     )-> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Postprocess the model outputs to extract class labels and bounding boxes.

        Args:
            outputs (np.ndarray): outputs from the model containing bounding boxes and class scores.
            original_shape (Tuple[int,int]): Original image shape (height, width).

        Returns:
            boxes (np.ndarray): Corresponding bounding boxes.
            classes (np.ndarray): Predicted class labels for each box.
            scores (np.ndarray): Confidence scores for each box.
            masks (np.ndarray): Segmentation masks for each box.
        """
        # boxes (1,200,4), logits (1,200,91), masks (1,200,108,108)
        boxes, logits, masks = outputs
        boxes = boxes[0]
        logits = logits[0]
        masks = masks[0]
        
        # Vectorized sigmoid computation
        all_scores = sigmoid(logits)  # (num_boxes, num_classes)
        max_scores = np.max(all_scores, axis=-1)  # (num_boxes,)
        
        # filtering by threshold
        thresh_mask = max_scores > self.thresh
        
        if not np.any(thresh_mask):
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Apply threshold filter first to reduce array sizes
        boxes = boxes[thresh_mask]
        max_scores = max_scores[thresh_mask]
        all_scores = all_scores[thresh_mask]
        masks = masks[thresh_mask]
        
        # Get labels and sort on filtered data
        labels = np.argmax(all_scores, axis=-1)
        sorted_idx = np.argsort(-max_scores)
        
        # Apply sorting
        boxes = boxes[sorted_idx]
        scores = max_scores[sorted_idx]
        masks = masks[sorted_idx]
        labels = labels[sorted_idx]

        boxes = xywh2xyxy(boxes)

        if len(masks) > 0:
            # Vectorized resize operation
            target_size = np.max(original_shape)
            interpolated_masks = []
            
            for mask in masks:
                # Direct resize without unnecessary normalization
                resized_mask = cv2.resize(mask.astype(np.float32), 
                            (target_size, target_size),
                            interpolation=cv2.INTER_LINEAR)
                # Direct boolean conversion with threshold
                mask_bool = (resized_mask > 0).astype(np.uint8)
                # Trim padded area
                mask_bool = mask_bool[:original_shape[0], :original_shape[1]]
                interpolated_masks.append(mask_bool)
        else:
            interpolated_masks = []
        
        masks = np.array(interpolated_masks) if interpolated_masks else np.array([])
    
        # denormalize boxes to original image size
        aspect = original_shape[1] / original_shape[0]
        if aspect >= 1:
            boxes[:, [0, 2]] *= original_shape[1]
            boxes[:, [1, 3]] *= original_shape[0]*aspect
        else:
            boxes[:, [0, 2]] *= original_shape[1]*aspect
            boxes[:, [1, 3]] *= original_shape[0]

        return boxes, labels, scores, masks
