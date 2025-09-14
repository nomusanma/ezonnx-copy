from typing import Any,Tuple,Dict,List,Union,Optional
import warnings
import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ezonnx.models.rtmdet.rtmdet import RTMDet
from ezonnx.data_classes.object_detection import ObjectDetectionResult,PoseDetectionResult
from ezonnx.ops.preprocess import standard_preprocess, image_from_path

class ViTPose(Inferencer):
    """ViTPose ONNX model for multi person pose estimation.
    For person detection, RTMDet "m-person" model will be used by default.

    Args:
        identifier (str): Model identifier, e.g., "small","base"
        quantize (Optional[str]): Quantization type, e.g., "fp16", "quantized". Default is None.
        person_detector (Optional[Inferencer]): Pre-trained person detector. If None, RTMDet "m-person" model will be used.
        kpt_thresh (float): Keypoint confidence threshold for filtering keypoints. Default is 0.3.
        iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS) in person detection. Default is 0.45.
        onnx_path (Optional[str]): Path to a local ONNX model file. If provided, the model will be loaded from this path instead of downloading. Default is None.
    
    Examples:
        Usage example:
        ::
            from ezonnx import ViTPose, visualize_images

            vitpose = ViTPose("small")  # you can choose "small","base" or pass your own model as onnx_path
            result = vitpose("image.jpg")

            print(result.boxes)  # (N, 4) array of bounding boxes
            print(result.kpts)  # (N,17,2) array of keypoints for each box
            print(result.scores)  # (N,) array of box confidence scores
            print(result.kpt_scores)  # (N,17) array of keypoint confidence scores
            print(result.visualized_img)  # (H, W, 3) image with keypoints drawn
    
    """

    def __init__(self,
                 identifier:str="small",
                 quantize:Optional[str]=None,
                 person_detector:Optional[Inferencer]=None,
                 kpt_thresh = 0.3,
                 iou_thresh = 0.45,
                 onnx_path:Optional[str]=None):
        # person detector
        if person_detector is None:
            self._person_det = RTMDet("m-person",iou_thresh=iou_thresh)
        else:
            self._person_det = person_detector
        
        # build
        self._check_backbone(identifier,["small","base"])
        self._check_quantize(quantize,[None,"fp16","quantized"])
         # Initialize model
        if onnx_path is None:
            # Initialize model
            repo_id = f"onnx-community/vitpose-plus-{identifier}-ONNX"
            filename = f"onnx/model.onnx"
            self.sess = self._download_and_compile(repo_id, filename,quantize)
        else:
            self.sess = self._compile_from_path(onnx_path)
        h,w = self.sess.get_inputs()[0].shape[2:]
        self.input_size_wh = (w,h)
        self.kpt_thresh = kpt_thresh

    def __call__(self,image:Union[str, np.ndarray])-> PoseDetectionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            PoseDetectionResult: Inference result containing boxes and classes.
        """
        image = image_from_path(image)
        # detect person
        det_result:ObjectDetectionResult = self._person_det(image)
        boxes = det_result.boxes
        box_scores = det_result.scores
        if len(boxes)==0:
            kpts = np.empty((0,17,2))
            kpt_scores = np.empty((0,17))

        for box in boxes:
            kpt,kpt_score = self.estimate_keypoints(box,image)
            if 'kpts' in locals():
                kpts = np.vstack([kpts,kpt])
            else:
                kpts = kpt
            if 'kpt_scores' in locals():
                kpt_scores = np.vstack([kpt_scores,kpt_score])
            else:
                kpt_scores = kpt_score

        return PoseDetectionResult(
            original_img = image,
            boxes = boxes,
            kpts = kpts,
            scores = box_scores,
            kpt_scores=kpt_scores,
            kpt_thresh = self.kpt_thresh
        )

    def _preprocess(self,
        img: np.ndarray, 
        input_size_wh: Tuple[int, int] = (192, 256),
        padding=1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            input_size_wh (tuple): Input image size in shape (w, h).

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        # get shape of image
        img_shape = img.shape[:2]
        bbox = np.array([0, 0, img_shape[1], img_shape[0]])

        # get center and scale
        center, scale = self._bbox_xyxy2cs(bbox, padding=padding)

        # do affine transformation
        resized_img, scale = self._top_down_affine(input_size_wh, scale, center, img)

        # normalize image
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std
        resized_img = np.transpose(resized_img, [2, 0, 1])
        resized_img = resized_img.astype("float32")[None, :]

        return resized_img, center, scale

    def estimate_keypoints(self,
                           box:np.ndarray,
                           frame:np.ndarray
                           )-> Tuple[np.ndarray,np.ndarray]:
        """Crop the person area and estimate keypoints.
        
        Args:
            box (np.ndarray): Bounding box in shape (4,), formatted as
                (left, top, right, bottom)
            frame (np.ndarray): Original image.

        Returns:
            tuple:
            - keypoints (np.ndarray): Estimated keypoints in shape (1, n_keypoints, 2).
            - scores (np.ndarray): Model predict scores of each key points.
        """
        frame_cropped = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
        # h, w = self.sess.get_inputs()[0].shape[2:]
        # model_input_size = (w, h)

        # preprocessing
        input_tensor, center, scale = self._preprocess(frame_cropped, self.input_size_wh)
        outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: input_tensor})
        # postprocessing
        keypoints, scores = self._postprocess(outputs[0],
                                              np.expand_dims(center,0),
                                              np.expand_dims(scale,0))
        # convert coordinates to original
        keypoints[0][:,0]+=int(box[0])
        keypoints[0][:,1]+=int(box[1])
        return keypoints,scores


    def _postprocess(self,heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):
        """Get final keypoint predictions from heatmaps and transform them back to
        the image.

        Note:
            - batch size: N
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
            center (np.ndarray[N, 2]): Center of the bounding box (x, y).
            scale (np.ndarray[N, 2]): Scale of the bounding box
                wrt height/width.
            post_process (str/None): Choice of methods to post-process
                heatmaps. Currently supported: None, 'default', 'unbiased',
                'megvii'.
            unbiased (bool): Option to use unbiased decoding. Mutually
                exclusive with megvii.
                Note: this arg is deprecated and unbiased=True can be replaced
                by post_process='unbiased'
                Paper ref: Zhang et al. Distribution-Aware Coordinate
                Representation for Human Pose Estimation (CVPR 2020).
            kernel (int): Gaussian kernel size (K) for modulation, which should
                match the heatmap gaussian sigma when training.
                K=17 for sigma=3 and k=11 for sigma=2.
            valid_radius_factor (float): The radius factor of the positive area
                in classification heatmap for UDP.
            use_udp (bool): Use unbiased data processing.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Classification target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
                Paper ref: Huang et al. The Devil is in the Details: Delving into
                Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

        Returns:
            tuple: A tuple containing keypoint predictions and scores.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        # Avoid being affected
        heatmaps = heatmaps.copy()

        # detect conflicts
        if unbiased:
            assert post_process not in [False, None, 'megvii']
        if post_process in ['megvii', 'unbiased']:
            assert kernel > 0
        if use_udp:
            assert not post_process == 'megvii'

        # normalize configs
        if post_process is False:
            warnings.warn(
                'post_process=False is deprecated, '
                'please use post_process=None instead', DeprecationWarning)
            post_process = None
        elif post_process is True:
            if unbiased is True:
                warnings.warn(
                    'post_process=True, unbiased=True is deprecated,'
                    " please use post_process='unbiased' instead",
                    DeprecationWarning)
                post_process = 'unbiased'
            else:
                warnings.warn(
                    'post_process=True, unbiased=False is deprecated, '
                    "please use post_process='default' instead",
                    DeprecationWarning)
                post_process = 'default'
        elif post_process == 'default':
            if unbiased is True:
                warnings.warn(
                    'unbiased=True is deprecated, please use '
                    "post_process='unbiased' instead", DeprecationWarning)
                post_process = 'unbiased'

        # start processing
        if post_process == 'megvii':
            heatmaps = self._gaussian_blur(heatmaps, kernel=kernel)

        N, K, H, W = heatmaps.shape
        if use_udp:
            if target_type.lower() == 'GaussianHeatMap'.lower():
                preds, maxvals = self._get_max_preds(heatmaps)
                preds = post_dark_udp(preds, heatmaps, kernel=kernel)
            elif target_type.lower() == 'CombinedTarget'.lower():
                for person_heatmaps in heatmaps:
                    for i, heatmap in enumerate(person_heatmaps):
                        kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                        cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
                # valid radius is in direct proportion to the height of heatmap.
                valid_radius = valid_radius_factor * H
                offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
                offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
                heatmaps = heatmaps[:, ::3, :]
                preds, maxvals = self._get_max_preds(heatmaps)
                index = preds[..., 0] + preds[..., 1] * W
                index += W * H * np.arange(0, N * K / 3)
                index = index.astype(int).reshape(N, K // 3, 1)
                preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
            else:
                raise ValueError('target_type should be either '
                                "'GaussianHeatmap' or 'CombinedTarget'")
        else:
            preds, maxvals = self._get_max_preds(heatmaps)
            if post_process == 'unbiased':  # alleviate biased coordinate
                # apply Gaussian distribution modulation.
                heatmaps = np.log(
                    np.maximum(self._gaussian_blur(heatmaps, kernel), 1e-10))
                for n in range(N):
                    for k in range(K):
                        preds[n][k] = self._taylor(heatmaps[n][k], preds[n][k])
            elif post_process is not None:
                # add +/-0.25 shift to the predicted locations for higher acc.
                for n in range(N):
                    for k in range(K):
                        heatmap = heatmaps[n][k]
                        px = int(preds[n][k][0])
                        py = int(preds[n][k][1])
                        if 1 < px < W - 1 and 1 < py < H - 1:
                            diff = np.array([
                                heatmap[py][px + 1] - heatmap[py][px - 1],
                                heatmap[py + 1][px] - heatmap[py - 1][px]
                            ])
                            preds[n][k] += np.sign(diff) * .25
                            if post_process == 'megvii':
                                preds[n][k] += 0.5

        # Transform back to the image
        for i in range(N):
            preds[i] = transform_preds(
                preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

        if post_process == 'megvii':
            maxvals = maxvals / 255.0 + 0.5
        return preds, maxvals
    
    def _get_max_preds(self,heatmaps):
        """Get keypoint predictions from score maps.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

        Returns:
            tuple: A tuple containing aggregated results.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        assert isinstance(heatmaps,
                        np.ndarray), ('heatmaps should be numpy.ndarray')
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals
    

    def _gaussian_blur(self, heatmaps, kernel=11):
        """Modulate heatmap distribution with Gaussian.
        sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
        sigma~=3 if k=17
        sigma=2 if k=11;
        sigma~=1.5 if k=7;
        sigma~=1 if k=3;

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
            kernel (int): Gaussian kernel size (K) for modulation, which should
                match the heatmap gaussian sigma when training.
                K=17 for sigma=3 and k=11 for sigma=2.

        Returns:
            np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
        """
        assert kernel % 2 == 1

        border = (kernel - 1) // 2
        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        height = heatmaps.shape[2]
        width = heatmaps.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmaps[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border),
                            dtype=np.float32)
                dr[border:-border, border:-border] = heatmaps[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmaps[i, j] = dr[border:-border, border:-border].copy()
                heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
        return heatmaps
    
    def _taylor(self, heatmap, coord):
        """Distribution aware coordinate decoding method.

        Note:
            - heatmap height: H
            - heatmap width: W

        Args:
            heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
            coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

        Returns:
            np.ndarray[2,]: Updated coordinates.
        """
        H, W = heatmap.shape[:2]
        px, py = int(coord[0]), int(coord[1])
        if 1 < px < W - 2 and 1 < py < H - 2:
            dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
            dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
            dxx = 0.25 * (
                heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
            dxy = 0.25 * (
                heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] -
                heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
            dyy = 0.25 * (
                heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] +
                heatmap[py - 2 * 1][px])
            derivative = np.array([[dx], [dy]])
            hessian = np.array([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = np.linalg.inv(hessian)
                offset = -hessianinv @ derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord
    


    def _bbox_xyxy2cs(self,bbox: np.ndarray,
                    padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
        """Transform the bbox format from (x,y,w,h) into (center, scale)

        Args:
            bbox (ndarray): Bounding box(es) in shape (4,) or (n, 4), formatted
                as (left, top, right, bottom)
            padding (float): BBox padding factor that will be multilied to scale.
                Default: 1.0

        Returns:
            tuple: A tuple containing center and scale.
            - np.ndarray[float32]: Center (x, y) of the bbox in shape (2,) or
                (n, 2)
            - np.ndarray[float32]: Scale (w, h) of the bbox in shape (2,) or
                (n, 2)
        """
        # convert single bbox from (4, ) to (1, 4)
        dim = bbox.ndim
        if dim == 1:
            bbox = bbox[None, :]

        # get bbox center and scale
        x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * padding

        if dim == 1:
            center = center[0]
            scale = scale[0]

        return center, scale


    def _fix_aspect_ratio(self,bbox_scale: np.ndarray,
                        aspect_ratio: float) -> np.ndarray:
        """Extend the scale to match the given aspect ratio.

        Args:
            scale (np.ndarray): The image scale (w, h) in shape (2, )
            aspect_ratio (float): The ratio of ``w/h``

        Returns:
            np.ndarray: The reshaped image scale in (2, )
        """
        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                            np.hstack([w, w / aspect_ratio]),
                            np.hstack([h * aspect_ratio, h]))
        return bbox_scale


    def _rotate_point(self,pt: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rotate a point by an angle.

        Args:
            pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
            angle_rad (float): rotation angle in radian

        Returns:
            np.ndarray: Rotated point in shape (2, )
        """
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt

    
    def _get_3rd_point(self,a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """To calculate the affine matrix, three pairs of points are required. This
        function is used to get the 3rd point, given 2D points a & b.

        The 3rd point is defined by rotating vector `a - b` by 90 degrees
        anticlockwise, using b as the rotation center.

        Args:
            a (np.ndarray): The 1st point (x,y) in shape (2, )
            b (np.ndarray): The 2nd point (x,y) in shape (2, )

        Returns:
            np.ndarray: The 3rd point.
        """
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c

    def _get_warp_matrix(self,
                        center: np.ndarray,
                        scale: np.ndarray,
                        rot: float,
                        output_size: Tuple[int, int],
                        shift: Tuple[float, float] = (0., 0.),
                        inv: bool = False) -> np.ndarray:
        """Calculate the affine transformation matrix that can warp the bbox area
        in the input image to the output size.

        Args:
            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            rot (float): Rotation angle (degree).
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            shift (0-100%): Shift translation ratio wrt the width/height.
                Default (0., 0.).
            inv (bool): Option to inverse the affine transform direction.
                (inv=False: src->dst or inv=True: dst->src)

        Returns:
            np.ndarray: A 2x3 transformation matrix
        """
        shift = np.array(shift)
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        # compute transformation matrix
        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        # get four corners of the src rectangle in the original image
        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        # get four corners of the dst rectangle in the input image
        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return warp_mat

    def _top_down_affine(self,
                        input_size: dict, bbox_scale: dict, bbox_center: dict,
                        img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bbox image as the model input by affine transform.

        Args:
            input_size (dict): The input size of the model.
            bbox_scale (dict): The bbox scale of the img.
            bbox_center (dict): The bbox center of the img.
            img (np.ndarray): The original image.

        Returns:
            tuple: A tuple containing center and scale.
            - np.ndarray[float32]: img after affine transform.
            - np.ndarray[float32]: bbox scale after affine transform.
        """
        w, h = input_size
        warp_size = (int(w), int(h))

        # reshape bbox to fixed aspect ratio
        bbox_scale = self._fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

        # get the affine matrix
        center = bbox_center
        scale = bbox_scale
        rot = 0
        warp_mat = self._get_warp_matrix(center, scale, rot, output_size=(w, h))

        # do affine transform
        img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

        return img, bbox_scale

def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    # scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords