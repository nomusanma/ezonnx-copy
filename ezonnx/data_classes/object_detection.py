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
    boxes: np.ndarray # (N, 4) or (N, 8) for OBB
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

class OBBResult(ObjectDetectionResult):
    """Data class for segmentation results.

    Attributes:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        boxes (np.ndarray): List of bounding boxes for each object in shape (8,). Default is None.
        classes (np.ndarray): List of class ids corresponding to each box. Default is None.
        scores (np.ndarray): List of confidence scores corresponding to each box. Default
        visualized_img (np.ndarray): Processed image with bounding boxes drawn.
    """
    angles: np.ndarray # (N,) range [-π/4, 3π/4]
    def _visualize(self) -> np.ndarray:
        """Get the processed image with segmentation masks applied.

        Returns:
            np.ndarray: Processed image in shape (H, W, 3). BGR
        """
        img = self.original_img.copy()
        line_width = max(2, int(min(img.shape[0], img.shape[1]) / 300))
        for box,cls,score,angle in zip(self.boxes,self.classes,self.scores,self.angles):
            # x1, y1, x2, y2, x3, y3, x4, y4 = box
            label_str = f"{cls} {score*100:.0f}%"
            color = COLORS[cls%100]
            pts = box.reshape(-1, 2).astype(int)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_width)
            
            # if draw angle line
            # center = np.mean(pts, axis=0).astype(int)
            # length = int(np.linalg.norm(pts[0]-pts[2])/2)
            # angle_rad = angle + np.pi/2
            # x2 = int(center[0] + length * np.cos(angle_rad))
            # y2 = int(center[1] + length * np.sin(angle_rad))
            # cv2.line(img, tuple(center), (x2, y2), color, thickness=line_width)

            # if put label
            # Draw label background
            # (text_width, text_height), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            # cv2.rectangle(img, (pts[0][0], pts[0][1] - text_height - baseline), 
            #               (pts[0][0] + text_width, pts[0][1]), color, -1)
            # # Put label text
            # cv2.putText(img, label_str, (pts[0][0], pts[0][1] - baseline), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img
    
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
            
        img = draw_boxes(img,self.boxes, self.classes,self.scores)

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
        line_width = max(2, int(min(img.shape[0], img.shape[1]) / 300))
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
        else:
            start_id = 0
            n_skelton = 0
        
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
                        draw_labels=False)

        return img
    

class FaceDetectionResult(Result):
    """Data class for pose detection results.

    Attributes:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        boxes (np.ndarray): List of bounding boxes for each object in shape (4,). Default is None.
        scores (np.ndarray): List of confidence scores corresponding to each box. Default
        kpts (np.ndarray): Array of keypoints in shape (N, num_keypoints, 2) where each keypoint is (x, y, confidence).
        kpt_scores (np.ndarray): Array of keypoint confidence scores in shape (N,num_keypoints). Default
        visualized_img (np.ndarray): Processed image with keypoints and skeleton drawn.
    """ 
    boxes: np.ndarray # (N, 4)
    
    scores: np.ndarray # (N,)
    kpts: np.ndarray # (N, num_keypoints, 2) arrays
    kpt_scores: np.ndarray #(N, num_keypoints)
    kpt_thresh: float
    face_im_size: tuple=(112,112)
    
    @property
    def face_imgs(self) -> List[np.ndarray]:
        """Extract and return aligned face images based on detected keypoints.

        Args:
            size (tuple, optional): Desired output size of the face images. Defaults to (112, 112).
        Returns:
            List[np.ndarray]: List of aligned face images in shape (H, W, 3). BGR
        """
        face_imgs = []
        for box in self.boxes:
            x1, y1, x2, y2 = box
            if x2<=x1 or y2<=y1:
                raise ValueError(f"Invalid box with zero area: {box}")
            face_imgs.append(self.original_img[int(y1):int(y2), int(x1):int(x2)])
        return face_imgs
        

    def _visualize(self) -> np.ndarray:
        """Visualize the keypoints and skeleton on image.

        Returns:
            img (np.ndarray): Visualized image.
        """
        img = self.original_img.copy()
        line_width = max(2, int(min(img.shape[0], img.shape[1]) / 200))
        keypoints = self.kpts
        scores = self.kpt_scores
        boxes = self.boxes
        thr = self.kpt_thresh
        # # default color
        n_keypoints = keypoints.shape[-2]
        # if n_keypoints==17:
        #     start_id = 0
        #     n_skelton = 19
        # elif n_keypoints==133:
        #     start_id = 0
        #     n_skelton = 65
        # elif n_keypoints==133:
        #     start_id = 0
        #     n_skelton = 65
        # elif n_keypoints==21:
        #     start_id = 25
        #     n_skelton = 44
        # else:
        #     start_id = 0
        #     n_skelton = 0
        # skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
        #             (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
        #             (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
        #             (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
        #             (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
        #             (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
        #             (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
        #             (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
        #             (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
        #             (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
        #             (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
        #             (129, 130), (130, 131), (131, 132)][start_id:n_skelton]
        # if n_keypoints==21:
        #     skeleton = np.array(skeleton) - 91

        palette = [[0, 0, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
                [255, 153, 255], [102, 178, 255], [255, 51, 51]]
        # link_color = [
        #     1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        #     2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        #     2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1][start_id:n_skelton]
        point_color = [
            0, 0, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4,
            4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ][:n_keypoints]

        # draw keypoints and skeleton
        for kpts, score, box in zip(keypoints, scores, boxes):
            kpts = kpts[:, :2] #if 3d pose, ignore z
            # 関節の信頼度の最大が0.3以下なら誤検出として表示しない
            if max(score)>0.3:
                for kpt, color,sc in zip(kpts, point_color,score):
                    if sc > thr:
                        cv2.circle(img, tuple(kpt.astype(np.int32)), line_width+2, palette[color], line_width,
                                cv2.LINE_AA)
                # for (u, v), color in zip(skeleton, link_color):
                #     if score[u] > thr and score[v] > thr:
                #         cv2.line(img, tuple(kpts[u].astype(np.int32)),
                #                 tuple(kpts[v].astype(np.int32)), palette[color], line_width,
                #                 cv2.LINE_AA)
        
        img = draw_boxes(img,
                        self.boxes, np.zeros((len(self.boxes),),dtype=int),self.scores,line_width=line_width,
                        draw_labels=False)

        return img