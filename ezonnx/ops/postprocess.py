import cv2
import numpy as np
from typing import Dict, Optional, Union, List

rng = np.random.default_rng(42)
COLORS = rng.uniform(0, 255, size=(100, 3))

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def draw_masks(image: np.ndarray, 
               masks: Dict[int, np.ndarray], 
               alpha: float = 0.5, 
               draw_border: bool = True
               ) -> np.ndarray:
    
    """Draw multiple masks on the image.

    Args:
        image (np.ndarray): Original image in shape (H, W, 3).
        masks (Dict[int, np.ndarray]): Dictionary of binary masks with label ids as keys.
        alpha (float): Transparency factor for the mask overlay. Default is 0.5.
        draw_border (bool): Whether to draw border around the masks. Default is True.
    
    Returns:
        np.ndarray: Image with the masks drawn on it.
    """
    mask_image = image.copy()

    for label_id, label_masks in masks.items():
        if label_masks is None:
            continue
        color = COLORS[label_id]
        mask_image = draw_mask(mask_image, label_masks, (color[0], color[1], color[2]), alpha, draw_border)

    return mask_image

def draw_mask(image: np.ndarray, 
              mask: np.ndarray, 
              color: tuple = (0, 255, 0), 
              alpha: float = 0.5, 
              draw_border: bool = True
              ) -> np.ndarray:
    """Draw a single mask on the image.

    Args:
        image (np.ndarray): Original image in shape (H, W, 3).
        mask (np.ndarray): Binary mask in shape (H, W) with values 0 or 1.
        color (tuple): Color for the mask in BGR format. Default is green (0, 255, 0).
        alpha (float): Transparency factor for the mask overlay. Default is 0.5.
        draw_border (bool): Whether to draw border around the mask. Default is True.
    
    Returns:
        np.ndarray: Image with the mask drawn on it.
    """

    mask_image = image.copy()
    mask_image[mask > 0.01] = color
    mask_image = cv2.addWeighted(image, 1-alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image

def draw_boxes(image: np.ndarray,
               boxes: np.ndarray,
               classes: np.ndarray,
               scores: np.ndarray,
               text_color: tuple = (255, 255, 255),
               draw_labels: bool = True
               ) -> np.ndarray:
    """Draw bounding boxes on the image.

    Args:
        image (np.ndarray): Original image in shape (H, W, 3).
        boxes (np.ndarray): Array of bounding boxes in shape (N, 4) with (x1, y1, x2, y2).
        classes (np.ndarray): Array of class ids corresponding to each box in shape (N,).
        scores (np.ndarray): Array of confidence scores corresponding to each box in shape (N,).
        box_color (tuple): Color for the bounding box in BGR format. Default is green (0, 255, 0).
        text_color (tuple): Color for the text in BGR format. Default is white (255, 255, 255).
        draw_labels (bool): Whether to draw class labels and scores. Default is True.
    
    Returns:
        np.ndarray: Image with the bounding boxes drawn on it.
    """
    box_image = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        box_color = COLORS[classes[i] % len(COLORS)]
        cv2.rectangle(box_image, (x1, y1), (x2, y2), box_color, thickness=2)
        if draw_labels:
            label = f"{classes[i]}: {scores[i]:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(box_image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), box_color, thickness=-1)
            cv2.putText(box_image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness=1)

    return box_image

def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    '''
    box and boxes are format as [x1, y1, x2, y2]
    '''
    # inter area
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, xmax-xmin) * np.maximum(0, ymax-ymin)

    # union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    return inter_area / union_area

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    sorted_idx = np.argsort(scores)[::-1]
    keep_idx = []
    while sorted_idx.size > 0:
        idx = sorted_idx[0]
        keep_idx.append(idx)
        ious = _compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])
        rest_idx = np.where(ious < iou_thr)[0]
        sorted_idx = sorted_idx[rest_idx+1]
    return keep_idx

def xywh2xyxy(box: np.ndarray) -> np.ndarray:
    box_xyxy = box.copy()
    box_xyxy[..., 0] = box[..., 0] - box[..., 2] / 2
    box_xyxy[..., 1] = box[..., 1] - box[..., 3] / 2
    box_xyxy[..., 2] = box[..., 0] + box[..., 2] / 2
    box_xyxy[..., 3] = box[..., 1] + box[..., 3] / 2
    return box_xyxy