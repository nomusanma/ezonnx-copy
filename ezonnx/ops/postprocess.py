import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union, List, Tuple
from scipy.signal import savgol_filter

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
               draw_labels: bool = True,
               line_width: Optional[int] = None
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
    line_width = max(2, int(min(image.shape[0], image.shape[1]) / 300)) if line_width is None else line_width
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        box_color = COLORS[classes[i] % len(COLORS)]
        cv2.rectangle(box_image, (x1, y1), (x2, y2), box_color, thickness=line_width)
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

def box_cxcywh_to_xyxy_numpy(x):
    x_c, y_c, w, h = np.split(x, 4, axis=-1)
    b = np.concatenate([
        x_c - 0.5 * np.clip(w, a_min=0.0, a_max=None),
        y_c - 0.5 * np.clip(h, a_min=0.0, a_max=None),
        x_c + 0.5 * np.clip(w, a_min=0.0, a_max=None),
        y_c + 0.5 * np.clip(h, a_min=0.0, a_max=None)
    ], axis=-1)
    return b

def apply_savgol_filter_to_skeleton(skeleton_data, window_length=5, polyorder=2):
    """
    時系列の2次元骨格データにSavitzky-Golayフィルターを適用する。
    
    Parameters:
    - skeleton_data: np.ndarray
        形状 (N, 17, 2) の時系列骨格データ
        N: フレーム数（時間軸）
        17: 関節の数
        2: [x, y] 座標
    - window_length: int
        フィルタのウィンドウサイズ（奇数である必要があります）
    - polyorder: int
        フィルタに使用される多項式の次数
    
    Returns:
    - filtered_data: np.ndarray
        Savitzky-Golayフィルターが適用された時系列データ
    """
    # 骨格データの形状を確認
    N, num_joints, num_coords = skeleton_data.shape
    assert num_joints == 17 and num_coords == 2, "入力データは形状 (N, 17, 2) である必要があります"
    
    # フィルタを適用するための出力配列を作成
    filtered_data = np.zeros_like(skeleton_data)
    
    # 各関節（17個）ごとに、x, y座標それぞれにSavitzky-Golayフィルターを適用
    for joint in range(num_joints):
        for coord in range(num_coords):
            # 各座標軸（xまたはy）にSavitzky-Golayフィルターを適用
            filtered_data[:, joint, coord] = savgol_filter(skeleton_data[:, joint, coord], 
                                                           window_length=window_length, 
                                                           polyorder=polyorder)
    
    return filtered_data

def draw_kpts(image: np.ndarray,
              kpts: np.ndarray,
              scores: Optional[np.ndarray] = None,
              kpt_thresh: Optional[float] = None,
              radius: int = -1,
              ) -> np.ndarray:
    """Draw keypoints on the image.

    Args:
        image (np.ndarray): Original image in shape (H, W, 3).BGR
        kpts (np.ndarray): Array of keypoints in shape (N, 2) with (x, y).
        scores (Optional[np.ndarray]): Array of confidence scores corresponding to each keypoint in shape (N,). Default is None.
        kpt_thresh (Optional[float]): Threshold to filter keypoints based on scores. Default is None.
        color_map (str): Color map for the keypoints. Default is 'jet'.
        radius (int): Radius of the keypoint circles. Default is -1 (auto).
    
    Returns:
        np.ndarray: BGR Image with the keypoints drawn on it.
    """
    kpt_image = image.copy()

    if kpt_image.ndim == 2:
        kpt_image = cv2.cvtColor(kpt_image, cv2.COLOR_GRAY2BGR)

    scores = scores if scores is not None else np.ones((kpts.shape[0],), dtype=np.float32)
    # radius is auto-calculated if set to -1
    radius = radius if radius > 0 else max(1, min(image.shape[0], image.shape[1]) // 100)
    for i, (kpt,score) in enumerate(zip(kpts,scores)):
        if kpt_thresh is not None and (scores is not None and scores[i] < kpt_thresh):
            continue
        kpt_color = (0,int(score*255),int((1-score)*255)) # BGR
        x, y = int(kpt[0]), int(kpt[1])
        cv2.circle(kpt_image, (x, y), radius, kpt_color, thickness=-1)
    return kpt_image
    
