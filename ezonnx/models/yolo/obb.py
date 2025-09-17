from typing import List,Dict,Optional,Union,Tuple

import numpy as np
from ...core.inferencer import Inferencer
from ...data_classes.object_detection import OBBResult
from ...ops.preprocess import (resize_with_aspect_ratio,
                                image_from_path,
                               standard_preprocess)

class YOLOOBB(Inferencer):
    """YOLOOBB model for object detection with ONNX.  
    Please export weights using ultralytics YOLOOBB.

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

        Example of exporting ONNX model from Ultralytics YOLOOBB
        ::
            from ultralytics import YOLO
            # Load the YOLO11 obb model
            model = YOLO("yolo11n-obb.pt")
            # Export the model to ONNX format
            model.export(format="onnx")  # creates 'yolo11n-obb.onnx'
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
                 )-> OBBResult:
        '''Run inference on the input image.
        
        Args:
            img: str or np.ndarray (H, W, C) BGR

        Returns:
            OBBResult: Inference result containing boxes, scores, classes, and visualized image
        '''
        img = image_from_path(img)
        tensor, ratio = self._preprocess(img)
        output = self.sess.run(None, {self.input_name: tensor})
        boxes, scores, classes,angles = self._postprocess(output, ratio)

        return OBBResult(original_img=img,
                                     boxes=boxes,
                                     scores=scores,
                                     classes=classes,
                                     angles=angles)
    
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
    
    def _postprocess(self, 
                     output: List[np.ndarray], 
                     ratio
                     ) -> Tuple[np.ndarray]:
        
        n_cls = output[0].shape[1] - 5
        output = output[0].squeeze(0).T
        angles = output[:, 4+n_cls:].squeeze(1)# range [-π/4, 3π/4]
        confs = np.max(output[:, 4:4+n_cls], axis=1)
        labels = np.argmax(output[:, 4:4+n_cls], axis=1)
        boxes = output[:, :4] # xc,yc,w,h
        # filter by conf_thresh
        boxes = boxes[confs > self.conf_thresh]
        labels = labels[confs > self.conf_thresh]
        angles = angles[confs > self.conf_thresh]
        confs = confs[confs > self.conf_thresh]
        # nms
        ids = nms_with_angles(boxes, confs, angles, self.iou_thresh)
        labels = labels[ids]
        confs = confs[ids]
        angles = angles[ids]
        boxes = boxes[ids] / ratio # rescale to original size
        # to xyxyxyxy
        boxes = get_obb_corners_xyxyxyxy(boxes, angles)
        return boxes, confs, labels, angles

def get_obb_corners_xyxyxyxy(boxes_xywh: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Get corners of oriented bounding boxes in xyxyxyxy format.
    
    Args:
        boxes_xywh (np.ndarray): Bounding boxes in xywh format (center_x, center_y, width, height)
        angles (np.ndarray): Rotation angles in radians for each box
        
    Returns:
        np.ndarray: Corner coordinates in xyxyxyxy format (x1,y1,x2,y2,x3,y3,x4,y4) for each box
    """
    if len(boxes_xywh) == 0:
        return np.array([]).reshape(0, 8)
    
    corners_list = []
    
    for box, angle in zip(boxes_xywh, angles):
        cx, cy, w, h = box
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Half dimensions
        hw, hh = w / 2, h / 2
        
        # Corner offsets relative to center (counter-clockwise from top-left)
        corners = np.array([
            [-hw, -hh],  # top-left
            [hw, -hh],   # top-right
            [hw, hh],    # bottom-right
            [-hw, hh]    # bottom-left
        ])
        
        # Rotate corners
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate to actual position
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        
        # Flatten to xyxyxyxy format
        corners_flat = rotated_corners.flatten()
        corners_list.append(corners_flat)
    
    return np.array(corners_list)

def nms_with_angles(boxes_xywh: np.ndarray, scores: np.ndarray, angles: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Non-Maximum Suppression for oriented bounding boxes.
    
    Args:
        boxes_xywh (np.ndarray): Bounding boxes in xywh format (center_x, center_y, width, height)
        scores (np.ndarray): Confidence scores for each box
        angles (np.ndarray): Rotation angles in radians for each box
        iou_threshold (float): IoU threshold for suppression
        
    Returns:
        np.ndarray: Indices of boxes to keep
    """
    if len(boxes_xywh) == 0:
        return np.array([], dtype=int)
    
    # Sort by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while len(sorted_indices) > 0:
        # Take the box with highest score
        current_idx = sorted_indices[0]
        keep.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes_xywh[current_idx]
        current_angle = angles[current_idx]
        
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes_xywh[remaining_indices]
        remaining_angles = angles[remaining_indices]
        
        # Calculate IoU for all remaining boxes
        ious = []
        for i, (box, angle) in enumerate(zip(remaining_boxes, remaining_angles)):
            iou = _calculate_obb_iou(current_box, current_angle, box, angle)
            ious.append(iou)
        
        ious = np.array(ious)
        
        # Keep boxes with IoU less than threshold
        mask = ious < iou_threshold
        sorted_indices = remaining_indices[mask]
    
    return np.array(keep, dtype=int)

def _calculate_obb_iou(box1_xywh: np.ndarray, angle1: float, box2_xywh: np.ndarray, angle2: float) -> float:
    """Calculate IoU between two oriented bounding boxes.
    
    Args:
        box1_xywh (np.ndarray): First box in xywh format
        angle1 (float): Rotation angle of first box in radians
        box2_xywh (np.ndarray): Second box in xywh format  
        angle2 (float): Rotation angle of second box in radians
        
    Returns:
        float: IoU value between 0 and 1
    """
    def get_obb_corners(cx, cy, w, h, angle):
        """Get corners of oriented bounding box."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Half dimensions
        hw, hh = w / 2, h / 2
        
        # Corner offsets relative to center
        corners = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])
        
        # Rotate corners
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate to actual position
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        
        return rotated_corners
    
    def polygon_area(vertices):
        """Calculate area of polygon using shoelace formula."""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0
    
    def line_intersection(p1, p2, p3, p4):
        """Find intersection point of two lines."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return [x, y]
    
    def sutherland_hodgman_clip(subject_polygon, clip_polygon):
        """Clip subject polygon by clip polygon using Sutherland-Hodgman algorithm."""
        def is_inside(point, edge_start, edge_end):
            return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - 
                    (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= 0
        
        output_list = list(subject_polygon)
        
        for i in range(len(clip_polygon)):
            if not output_list:
                break
                
            input_list = output_list
            output_list = []
            
            edge_start = clip_polygon[i]
            edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
            
            if input_list:
                s = input_list[-1]
                
                for e in input_list:
                    if is_inside(e, edge_start, edge_end):
                        if not is_inside(s, edge_start, edge_end):
                            intersection = line_intersection(s, e, edge_start, edge_end)
                            if intersection:
                                output_list.append(intersection)
                        output_list.append(e)
                    elif is_inside(s, edge_start, edge_end):
                        intersection = line_intersection(s, e, edge_start, edge_end)
                        if intersection:
                            output_list.append(intersection)
                    s = e
        
        return output_list
    
    # Get corners of both boxes
    corners1 = get_obb_corners(box1_xywh[0], box1_xywh[1], box1_xywh[2], box1_xywh[3], angle1)
    corners2 = get_obb_corners(box2_xywh[0], box2_xywh[1], box2_xywh[2], box2_xywh[3], angle2)
    
    # Calculate intersection using polygon clipping
    intersection_polygon = sutherland_hodgman_clip(corners1, corners2)
    
    if len(intersection_polygon) < 3:
        return 0.0
    
    # Calculate areas
    intersection_area = polygon_area(intersection_polygon)
    area1 = box1_xywh[2] * box1_xywh[3]
    area2 = box2_xywh[2] * box2_xywh[3]
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area



