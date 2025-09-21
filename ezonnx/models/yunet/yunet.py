from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np

from ezonnx.core.inferencer import Inferencer
from ...data_classes.object_detection import FaceDetectionResult
from ...core.downloader import get_weights
from ...ops.preprocess import image_from_path, letterbox
from ...ops.postprocess import xlylwh2xyxy

class YuNet(Inferencer):
    """YuNet ONNX model for face detection and landmark estimation.

    Args:
        quantize (Optinal[str]): "int8" or None
        size (int): Input size for the model. Default is 640.
        conf_thresh (float): Confidence threshold for detections. Default is 0.5.
        iou_thresh (float): IoU threshold for NMS. Default is 0.45.
        topk (int): Maximum number of detections to keep. Default is 200.
        onnx_path (Optional[str]): Path to the ONNX model file. If None, the model will be downloaded.
    
    Examples:
        Usage example:
        ::
            from ezonnx import YuNet

            net = YuNet()
            result = net("image.jpg")

            print(result.boxes)  # (N, 4) array of bounding boxes
            print(result.kpts)  # (N,5,2) array of keypoints for each box
            print(result.scores)  # (N,) array of confidence scores
            print(result.kpt_scores)  # (N,17) array of keypoint confidence scores
            print(result.visualized_img)  # (H, W, 3) image with
    
    """

    def __init__(
        self,
        quantize:Optional[str]=None,
        size:int=640,
        conf_thresh:float=0.3,
        iou_thresh:float=0.3,
        topk=500,
        onnx_path:Optional[str]=None
    ):
        if onnx_path is None:
            self._check_quantize(quantize,["int8",None])
            # Initialize model
            repo_id = f"bukuroo/YuNet-ONNX"
            filename = f"yunet.onnx"
            if quantize is not None:
                filename = filename.replace(".onnx",f"_{quantize}.onnx")
            onnx_path = get_weights(repo_id, filename)
        self._model = cv2.FaceDetectorYN.create(onnx_path, "", 
                                          (size, size),
                                          score_threshold = conf_thresh,
                                          top_k=topk,
                                          nms_threshold=iou_thresh)
        # 各種設定
        self.size = size

    def __call__(self,image:Union[str, np.ndarray])-> FaceDetectionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            PoseDetectionResult: Inference result containing boxes and keypoints.
        """
        image = image_from_path(image)
        padded_img,ratio = self._preprocess(image)
        _,faces= self._model.detect(padded_img)
        boxes, kpts, scores = self._postprocess(faces,
                                                ratio)

        return FaceDetectionResult(
            original_img=image,
            boxes=boxes,
            scores=scores,
            kpts=kpts,
            kpt_scores=np.ones((len(boxes),5),dtype=np.float32),
            kpt_thresh=0
        )
    
    def _preprocess(self,image:np.ndarray
                    )-> Tuple[np.ndarray,float]:
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """
        return letterbox(image,self.size)

    def _postprocess(self,
        faces: Optional[np.ndarray],
        ratio: float
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        faces: (N, 15) = [x,y,w,h, 10*kpts(xy), score]
        ratio: letterbox比。
        """
        if faces is None or len(faces) == 0:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0, 5, 2), np.float32),
                    np.zeros((0,), np.float32))

        faces = np.ascontiguousarray(faces, dtype=np.float32)
        n = faces.shape[0]
        inv = 1.0 / float(ratio)

        # --- boxes ---

        boxes = xlylwh2xyxy(faces[:, :4])
        np.multiply(boxes, inv, out=boxes)

        # --- kpts (N,5,2) ---
        kpts = faces[:, 4:14].reshape(n, 5, 2).copy() 
        np.multiply(kpts, inv, out=kpts)

        # --- scores (N,) ---
        scores = faces[:, 14].astype(np.float32, copy=False)

        return boxes, kpts, scores
    