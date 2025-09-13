from typing import List,Dict,Optional,Union,Tuple

import numpy as np
from ...core.inferencer import Inferencer
from ...data_classes.object_detection import ObjectDetectionResult
from ...ops.preprocess import (resize_with_aspect_ratio,
                                image_from_path,
                               standard_preprocess)
from ...ops.postprocess import nms,xywh2xyxy

class YOLO(Inferencer):
    """YOLO model for object detection with ONNX.

    Args:
        onnx_path (str): Path to a local ONNX model file.
        conf_thresh (float): Confidence threshold for filtering detections. Default is 0.3.
        iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS). Default is 0.45.
    
    Examples:
        Usage
        ::
            from ezonnx import YOLO, visualize_images
            det = YOLO("/path/to/yolo.onnx") # Please use local weight
            ret = det("images/surf.jpg")
            visualize_images("Detection Result",ret.visualized_img)

    Please use ONNX weight exported from ultralytics library.
    Example of exporting to ONNX:
        ::
            from ultralytics import YOLO
            # Load the YOLO11 model
            model = YOLO("yolo11n.pt")
            # Export the model to ONNX format
            model.export(format="onnx")  # creates 'yolo11n.onnx'
    """

    def __init__(self,
                 onnx_path: str,
                 conf_thresh=0.25,
                 iou_thresh=0.45
                 ) -> None:
        if onnx_path is None:
            raise ValueError("Please provide the onnx_path for YOLOSeg model.Remote repo not available yet.")
            # self._check_backbone(identifier,[""])
            # self._check_quantize(quantize,
            #                     [None])
            # # Initialize model
            # repo_id = f""
            # filename = f""
            # self.sess = self._download_and_compile(repo_id, filename, quantize)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.size = self.sess.get_inputs()[0].shape[3] # width
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def __call__(self,
                 img:Union[str, np.ndarray]
                 )-> ObjectDetectionResult:
        '''Run inference on the input image.
        
        Args:
            img: str or np.ndarray (H, W, C) BGR

        Returns:
            ObjectDetectionResult: Inference result containing boxes, scores, classes, and visualized image
        '''
        img = image_from_path(img)
        tensor, ratio = self._preprocess(img)
        output = self.sess.run(None, {self.input_name: tensor})
        boxes, scores, classes = self._postprocess(output, ratio)

        return ObjectDetectionResult(original_img=img,
                                     boxes=boxes,
                                     scores=scores,
                                     classes=classes)
    
    def _preprocess(self, img: np.ndarray):
        ''' 
        img(np.ndarray) cv2image BGR HWC
        '''
        """Preprocess the input image for the model.

        Args:
            img (np.ndarray): Input image array.

        Returns:
            Tuple[np.ndarray,float]: Preprocessed image tensor in shape (1, 3, H, W) and resize ratio.
        """
        padded_img,r = resize_with_aspect_ratio(img,self.size)
        tensor = standard_preprocess(padded_img,
                                    size=None,
                                    standardize=False)
        return tensor, r
    
    def _postprocess(self, output: List[np.ndarray], ratio
                     ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        predict = output[0].squeeze(0).T
        predict = predict[np.max(predict[:, 4:],axis=1) > self.conf_thresh, :]
        scores = np.max(predict[:, 4:],axis=1)
        classes = np.argmax(predict[:, 4:],axis=1)

        boxes = predict[:, 0:4] / ratio
        boxes = xywh2xyxy(boxes)

        idxes = nms(boxes, scores, self.iou_thresh)

        boxes= boxes[idxes,: ].astype(int)
        scores=scores[idxes]
        classes=classes[idxes]
        return boxes, scores, classes

