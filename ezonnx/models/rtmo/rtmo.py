from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ...data_classes.object_detection import PoseDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path, resize_with_aspect_ratio
from ...ops.postprocess import nms

class RTMO(Inferencer):
    """RTMO ONNX model for multi person detection and pose estimation.

    Args:
        identifier (str): Model identifier, e.g., "n","s","m","l"
        thresh (float): Confidence threshold for filtering detections. Default is 0.3.
        size (int): Input image size for the model. Default is 640. must be a multiple of 16.
        iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.45.
        onnx_path (Optional[str]): Path to a local ONNX model file. If provided, the model will be loaded from this path instead of downloading. Default is None.
    
    Examples:
        Usage example:
        ::
            from ezonnx import RTMO

            rtmo = RTMO("s")  # you can choose "n","s","m","l"
            result = rtmo("image.jpg")

            print(result.boxes)  # (N, 4) array of bounding boxes
            print(result.kpts)  # (N,17,2) array of keypoints for each box
            print(result.scores)  # (N,) array of confidence scores
            print(result.kpt_scores)  # (N,17) array of keypoint confidence scores
            print(result.visualized_img)  # (H, W, 3) image with
    
    """

    def __init__(self,
                 identifier=None,
                 thresh:float=0.5,
                 kpt_thresh:float=0.45,
                 size=640,
                 onnx_path:Optional[str]=None):
        
        if onnx_path is None:
            self._check_backbone(identifier,["n","s","m","l"])
            # Initialize model
            repo_id = f"bukuroo/RTMO-ONNX"
            filename = f"rtmo-{identifier}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = size
        self.thresh = thresh
        self.kpt_thresh = kpt_thresh
    
    def __call__(self,image:Union[str, np.ndarray])-> PoseDetectionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            PoseDetectionResult: Inference result containing boxes and keypoints.
        """
        image = image_from_path(image)
        input_tensor,ratio = self._preprocess(image)
        outputs = self.sess.run(None,
                            {self.input_name: input_tensor})
        boxes, kpts, scores,kpt_scores = self._postprocess(outputs,ratio)

        return PoseDetectionResult(
            original_img=image,
            boxes=boxes,
            scores=scores,
            kpts=kpts,
            kpt_scores=kpt_scores,
            kpt_thresh=self.kpt_thresh
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
        padded_img, ratio= resize_with_aspect_ratio(image,self.size)
        # rtmo model not require normalization
        input_tensor = np.expand_dims(padded_img.transpose((2, 0, 1))[::-1, ],axis=0)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
        return input_tensor, ratio
    
    def _postprocess(self,
                     outputs:np.ndarray,
                     ratio:float
                     )-> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Postprocess the model outputs to extract class labels and bounding boxes.

        Args:
            outputs (np.ndarray): outputs from the model containing bounding boxes and keypoints.
            ratio (float): Scaling ratio used during preprocessing to resize the image.
        
        Returns:
            boxes (np.ndarray): Corresponding bounding boxes.
            kpts (np.ndarray): Corresponding keypoints.
            scores (np.ndarray): Confidence scores for each detection.
            kpt_scores (np.ndarray): Confidence scores for each keypoint.
        """
        # separate boxes, scores
        boxes_scores = outputs[0][0]
        boxes = boxes_scores[:, :4]
        scores = boxes_scores[:, 4]
        # separate kpts, kpts_scores
        kpts_scores = outputs[1][0]
        kpts = kpts_scores[:, :, :2]
        kpt_scores = kpts_scores[:,:, 2]

        # filter by score threshold
        boxes_filtered = boxes[scores>=self.thresh]
        scores_filtered = scores[scores>=self.thresh]
        kpts_filtered = kpts[scores>=self.thresh]
        kpt_scores_filtered = kpt_scores[scores>=self.thresh]

        # scale to original image
        boxes_filtered = np.array(boxes_filtered) / ratio
        kpts_filtered = np.array(kpts_filtered) / ratio

        return boxes_filtered, kpts_filtered, scores_filtered,kpt_scores_filtered