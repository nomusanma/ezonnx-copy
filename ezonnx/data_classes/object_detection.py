from typing import List, Optional, Dict
import cv2
import numpy as np
from .result import Result

from ..ops.postprocess import draw_boxes,COLORS

class ObjectDetectionResult(Result):
    """Data class for segmentation results.

    Attributes:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        boxes (np.ndarray): List of bounding boxes for each object in shape (4,). Default is None.
        classes (np.ndarray): List of class ids corresponding to each box. Default is None.
        scores (np.ndarray): List of confidence scores corresponding to each box. Default
        visualized_img (np.ndarray): Processed image with bounding boxes drawn.
    """
    boxes: np.ndarray # (N, 4)
    classes: np.ndarray # (N,)
    scores: np.ndarray # (N,)

    def _visualize(self) -> np.ndarray:
        """Get the processed image with segmentation masks applied.

        Returns:
            np.ndarray: Processed image in shape (H, W, 3). BGR
        """
        return draw_boxes(self.original_img,
                        self.boxes, self.classes,self.scores,
                        draw_labels=True)
    
class InstanceSegmentationResult(Result):
    """Data class for segmentation results.

    Attributes:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        boxes (np.ndarray): List of bounding boxes for each object in shape (4,). Default is None.
        classes (np.ndarray): List of class ids corresponding to each box. Default is None.
        scores (np.ndarray): List of confidence scores corresponding to each box. Default
        masks (np.ndarray): Array of binary masks in shape (N, H, W) where N is the number of detected objects.
        visualized_img (np.ndarray): Processed image with masks and bounding boxes drawn.
    """
    boxes: np.ndarray # (N, 4)
    classes: np.ndarray # (N,)
    scores: np.ndarray # (N,)
    masks: np.ndarray # (N, H, W)

    def _visualize(self) -> np.ndarray:
        """Get the processed image with segmentation masks applied.

        Returns:
            np.ndarray: Processed image in shape (H, W, 3). BGR
        """
        img = self.original_img.copy()

        for box, mask, score,class_num in zip(self.boxes, self.masks, self.scores,self.classes):
            x1, y1, x2, y2 = box
            label_str = f"{class_num} {score*100:.0f}%"
            color = COLORS[class_num%20]

            img = img.astype(np.float32)
            img[mask==1] /= 2
            img[mask==1] += (np.array(COLORS[class_num%20]) / 2)

            label_size, baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2) 
            cv2.rectangle(img, (x1, y1), (x1+label_size[0], y1+label_size[1]+baseline),
                color, -1) #poseモデルの場合color固定
            cv2.putText(img, label_str, (x1, y1+label_size[1]), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

        return img.astype(np.uint8)
    

class PoseDetectionResult(Result):
    """Data class for pose detection results.

    Attributes:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        boxes (np.ndarray): List of bounding boxes for each object in shape (4,). Default is None.
        scores (np.ndarray): List of confidence scores corresponding to each box. Default
        kpts (np.ndarray): Array of keypoints in shape (N, num_keypoints, 3) where each keypoint is (x, y, confidence).
        kpt_scores (np.ndarray): Array of keypoint confidence scores in shape (N,num_keypoints). Default
        visualized_img (np.ndarray): Processed image with keypoints and skeleton drawn.
    """ 
    boxes: np.ndarray # (N, 4)
    scores: np.ndarray # (N,)
    kpts: np.ndarray # (N, num_keypoints, 3) arrays
    kpt_scores: np.ndarray #(N, num_keypoints)
    kpt_thresh: float

    def _visualize(self) -> np.ndarray:
        """Visualize the keypoints and skeleton on image.

        Returns:
            img (np.ndarray): Visualized image.
        """
        img = self.original_img.copy()
        keypoints = self.kpts
        scores = self.kpt_scores
        boxes = self.boxes
        thr = self.kpt_thresh
        # default color
        n_keypoints = keypoints.shape[-2]
        if n_keypoints==17:
            start_id = 0
            n_skelton = 19
        elif n_keypoints==133:
            start_id = 0
            n_skelton = 65
        elif n_keypoints==133:
            start_id = 0
            n_skelton = 65
        elif n_keypoints==21:
            start_id = 25
            n_skelton = 44
        skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                    (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                    (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                    (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                    (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                    (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                    (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                    (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                    (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                    (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                    (129, 130), (130, 131), (131, 132)][start_id:n_skelton]
        if n_keypoints==21:
            skeleton = np.array(skeleton) - 91

        palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
                [255, 153, 255], [102, 178, 255], [255, 51, 51]]
        link_color = [
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
            2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
            2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1][start_id:n_skelton]
        point_color = [
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4,
            4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ][:n_keypoints]

        # draw keypoints and skeleton
        for kpts, score,box in zip(keypoints, scores, boxes):
            kpts = kpts[:, :2] #if 3d pose, ignore z
            line_width = int(box[2]-box[0])//100
            # 関節の信頼度の最大が0.3以下なら誤検出として表示しない
            if max(score)>0.3:
                for kpt, color,sc in zip(kpts, point_color,score):
                    if sc > thr:
                        cv2.circle(img, tuple(kpt.astype(np.int32)), line_width+2, palette[color], line_width,
                                cv2.LINE_AA)
                for (u, v), color in zip(skeleton, link_color):
                    if score[u] > thr and score[v] > thr:
                        cv2.line(img, tuple(kpts[u].astype(np.int32)),
                                tuple(kpts[v].astype(np.int32)), palette[color], line_width,
                                cv2.LINE_AA)
        
        img = draw_boxes(img,
                        self.boxes, np.zeros((len(self.boxes),),dtype=int),self.scores,
                        draw_labels=True,line_width=line_width)

        return img