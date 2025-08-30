from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ...data_classes.object_detection import ObjectDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path, resize_with_aspect_ratio
from ...ops.postprocess import nms

class RTMDet(Inferencer):
    """RTMDet ONNX model for object detection.

    Args:
        identifier (str): Model identifier, e.g., "n-person","m-person","n-hand",
                                    "tiny-coco","s-coco",
                                    "m-coco","l-coco","x-coco".
        thresh (float): Confidence threshold for filtering detections. Default is 0.3.
        size (int): Input image size for the model. Default is 640. must be a multiple of 16.
        iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.45.
        onnx_path (Optional[str]): Path to a local ONNX model file. If provided, the model will be loaded from this path instead of downloading. Default is None.
    
    Examples:
        Usage example:
        ::
            from ezonnx import RTMDet

            rtm = RTMDet("s-coco")  # you can choose "n-person","m-person","n-hand",
                                    "tiny-coco","s-coco",
                                    "m-coco","l-coco","x-coco"
            result = rtm("image.jpg")

            print(result.boxes)  # (N, 4) array of bounding boxes
            print(result.classes)  # (N,) array of class labels
            print(result.scores)  # (N,) array of confidence scores
            print(result.visualized_img)  # (H, W, 3) image with
    
    """

    def __init__(self,
                 identifier=None,
                 thresh:float=0.3,
                 iou_thresh:float=0.45,
                 size=640,
                 onnx_path:Optional[str]=None):
        
        if onnx_path is None:
            self._check_backbone(identifier,["n-person","m-person","n-hand",
                                             "tiny-coco","s-coco",
                                             "m-coco","l-coco","x-coco"])
            # Initialize model
            repo_id = f"bukuroo/RTMDet-ONNX"
            filename = f"rtmdet-{identifier}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size
        self.thresh = thresh
        self.iou_thresh = iou_thresh
    
    def __call__(self,image:Union[str, np.ndarray])-> ObjectDetectionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            ObjectDetectionResult: Inference result containing boxes and classes.
        """
        image = image_from_path(image)
        input_tensor,ratio = self._preprocess(image)
        outputs = self.sess.run(None,
                            {self.input_name: input_tensor})
        boxes, classes, scores = self._postprocess(outputs,ratio)

        return ObjectDetectionResult(
            original_img=image,
            boxes=boxes,
            classes=classes,
            scores=scores
        )
    
    def _preprocess(self,image:np.ndarray
                    )-> Tuple[np.ndarray,float]:
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded_image, ratio= resize_with_aspect_ratio(image,self.size)
        # normalize image, rtmdet uses 0-255 range with mean/std
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        tensor = (padded_image - mean) / std
        tensor = np.transpose(tensor,[2,0,1])
        return tensor[None,:].astype("float32"), ratio
    
    def _postprocess(self,
                     outputs:np.ndarray,
                     ratio:float
                     )-> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Postprocess the model outputs to extract class labels and bounding boxes.

        Args:
            outputs (np.ndarray): outputs from the model containing bounding boxes and class scores.

        Returns:
            boxes (np.ndarray): Corresponding bounding boxes.
            classes (np.ndarray): Predicted class labels for each box.
            scores (np.ndarray): Confidence scores for each box.
        """
        # separate boxes, scores, classes
        boxes = outputs[0][0][:,:4] # unit is pixel in resized image
        scores = outputs[0][0][:,4]
        classes = outputs[1][0]

        # filter by score threshold
        boxes = boxes[scores>=self.thresh]
        classes = classes[scores>=self.thresh]
        scores = scores[scores>=self.thresh]

        # nms
        keep_id = nms(boxes,
                    scores,
                    iou_thr=self.iou_thresh)
        
        boxes = boxes[keep_id]
        classes = classes[keep_id]
        scores = scores[keep_id]
        
        # scale to original image
        boxes = np.array(boxes) / ratio

        return boxes, classes, scores