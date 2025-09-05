import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import List
from .result import Result

class BoxedResult(BaseModel):
    box: np.ndarray
    img: np.ndarray
    text: str
    score:float
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OCRResult(Result):
    """Data class for optical flow results.

    Args:
        original_img (np.ndarray): Original input image in shape (H, W, 3). BGR
        flow (np.ndarray): Optical flow in shape (H, W, 2). dtype=float32
        magnitude (np.ndarray): Magnitude of the optical flow in shape (H, W). dtype=float32
        angle (np.ndarray): Angle of the optical flow in shape (H, W). dtype=float32
        visualized_img (np.ndarray): Processed image with optical flow drawn.(H, W, 3). BGR
    """
    boxed_results: List[BoxedResult]

    def _visualize(self) -> np.ndarray:
        """Return the RGB representation of the optical flow.

        Returns:
            np.ndarray: RGB representation of the optical flow in shape (H, W, 3). BGR
        """
        img = np.ones(self.original_img.shape, dtype=np.uint8) * 255
        # 画像の最外周を2pxの黒線で囲う
        cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (0, 0, 0), 2)
        for box in self.boxed_results:
            points = box.box.astype(int).reshape((-1,1,2))
            box_width = int(np.linalg.norm(box.box[0]-box.box[1]))
            text_width,text_height = cv2.getTextSize(box.text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            font_scale = box_width / text_width[0]
            cv2.polylines(img,[points],True,(0,255,0),2)
            # テキストのサイズを取得
            (text_w, text_h), baseline = cv2.getTextSize(box.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            # ボックスの上下方向の中心を計算
            y_top = np.min(points[:,0,1])
            y_bottom = np.max(points[:,0,1])
            y_center = (y_top + y_bottom) // 2
            # テキストのベースラインがボックスの中心に来るようにy座標を調整
            text_y = y_center + text_h // 2
            # 左端のx座標
            text_x = np.min(points[:,0,0])
            cv2.putText(img, box.text, 
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (45,45,45), 2)

        return img

