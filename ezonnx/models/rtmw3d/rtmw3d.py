from typing import Any,Tuple,Dict,List,Union,Optional
from itertools import product
import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ..rtmdet.rtmdet import RTMDet
from ..rtmpose.rtmpose import RTMPose
from ...data_classes.object_detection import ObjectDetectionResult,PoseDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path

class RTMW3D(RTMPose):
    def __init__(self,
                 identifier:str="x-384",
                 person_detector:Inferencer=None,
                 kpt_thresh = 0.3,
                 onnx_path:Optional[str]=None):
        # person detector
        if person_detector is None:
            self._person_det = RTMDet("m-person")
        else:
            self._person_det = person_detector
        
        # build
        self._check_backbone(identifier,["x-384"])
        if onnx_path is None:
            # Initialize model
            repo_id = f"bukuroo/RTMW3D-ONNX"
            filename = f"rtmw3d-{identifier}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)
        self.input_size = self.sess.get_inputs()[0].shape[2:]
        self.input_size_wh = self.input_size[::-1]
        self.kpt_thresh = kpt_thresh

                # Mean value of the root z-axis of datasets
        # These values are statistics from the training set
        self.root_z = [5.14388]
        self.z_range = 2.1744869

    # def estimate_keypoints(self,
    #                        box:np.ndarray,
    #                        frame:np.ndarray
    #                        )-> Tuple[np.ndarray,np.ndarray]:
    #     """Crop the person area and estimate keypoints.
        
    #     Args:
    #         box (np.ndarray): Bounding box in shape (4,), formatted as
    #             (left, top, right, bottom)
    #         frame (np.ndarray): Original image.

    #     Returns:
    #         tuple:
    #         - keypoints (np.ndarray): Estimated keypoints in shape (1, n_keypoints, 2).
    #         - scores (np.ndarray): Model predict scores of each key points.
    #     """
    #     frame_cropped = frame[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:]
    #     # h, w = self.sess.get_inputs()[0].shape[2:]
    #     # model_input_size = (w, h)

    #     # preprocessing
    #     input_tensor, center, scale = self._preprocess(frame_cropped, self.input_size_wh)
    #     outputs = self.sess.run(None, {self.sess.get_inputs()[0].name: input_tensor})
    #     # postprocessing
    #     keypoints, scores = self._postprocess(outputs, self.input_size_wh, center, scale)
    #     # convert coordinates to original
    #     keypoints[0][:,0]+=int(box[0])
    #     keypoints[0][:,1]+=int(box[1])
    #     return keypoints,scores
    
    def _postprocess(self,
                    outputs: List[np.ndarray],
                    model_input_size: Tuple[int, int],
                    center: Tuple[int, int],
                    scale: Tuple[int, int],
                    simcc_split_ratio: float = 2.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # use simcc to decode
        simcc_x, simcc_y, simcc_z = outputs
        keypoints, keypoints_simcc, scores = self._decode(simcc_x, simcc_y, simcc_z, simcc_split_ratio)

        # rescale keypoints
        keypoints[:,:,:2] = keypoints[:,:,:2] / model_input_size * scale + center - scale / 2

        return keypoints, scores
    


    def _get_simcc_maximum(self,
                        simcc_x: np.ndarray,
                        simcc_y: np.ndarray,
                        simcc_z: np.ndarray,
                        apply_softmax: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Get maximum response location and value from simcc representations.

        Note:
            instance number: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            encoded_keypoints (dict): encoded keypoints with simcc representations.
            apply_softmax (bool): whether to apply softmax on the heatmap.
                Defaults to False.

        Returns:
            tuple:
            - locs (np.ndarray): locations of maximum heatmap responses in shape
                (K, 2) or (N, K, 2)
            - vals (np.ndarray): values of maximum heatmap responses in shape
                (K,) or (N, K)
        """
        assert isinstance(simcc_x, np.ndarray), 'simcc_x should be numpy.ndarray'
        assert isinstance(simcc_y, np.ndarray), 'simcc_y should be numpy.ndarray'
        assert isinstance(simcc_z, np.ndarray), 'simcc_z should be numpy.ndarray'
        assert simcc_x.ndim == 2 or simcc_x.ndim == 3, (
            f'Invalid shape {simcc_x.shape}')
        assert simcc_y.ndim == 2 or simcc_y.ndim == 3, (
            f'Invalid shape {simcc_y.shape}')
        assert simcc_z.ndim == 2 or simcc_z.ndim == 3, (
            f'Invalid shape {simcc_z.shape}')
        assert simcc_x.ndim == simcc_y.ndim == simcc_z.ndim, (
            f'{simcc_x.shape} != {simcc_y.shape} or {simcc_z.shape}')

        if simcc_x.ndim == 3:
            n, k, _ = simcc_x.shape
            simcc_x = simcc_x.reshape(n * k, -1)
            simcc_y = simcc_y.reshape(n * k, -1)
            simcc_z = simcc_z.reshape(n * k, -1)
        else:
            n = None

        if apply_softmax:
            simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
            simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
            simcc_z = simcc_z - np.max(simcc_z, axis=1, keepdims=True)
            ex, ey, ez = np.exp(simcc_x), np.exp(simcc_y), np.exp(simcc_z)
            simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
            simcc_y = ey / np.sum(ey, axis=1, keepdims=True)
            simcc_z = ez / np.sum(ez, axis=1, keepdims=True)

        x_locs = np.argmax(simcc_x, axis=1)
        y_locs = np.argmax(simcc_y, axis=1)
        z_locs = np.argmax(simcc_z, axis=1)
        locs = np.stack((x_locs, y_locs, z_locs), axis=-1).astype(np.float32)
        max_val_x = np.amax(simcc_x, axis=1)
        max_val_y = np.amax(simcc_y, axis=1)

        mask = max_val_x > max_val_y
        max_val_x[mask] = max_val_y[mask]
        vals = max_val_x
        locs[vals <= 0.] = -1

        if n is not None:
            locs = locs.reshape(n, k, 3)
            vals = vals.reshape(n, k)

        return locs, vals

    # def _decode(self,simcc_x: np.ndarray,
    #              simcc_y: np.ndarray,
    #              simcc_z: np.ndarray,
    #         simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    #     """Modulate simcc distribution with Gaussian.

    #     Args:
    #         simcc_x (np.ndarray[K, Wx]): model predicted simcc in x.
    #         simcc_y (np.ndarray[K, Wy]): model predicted simcc in y.
    #         simcc_z (np.ndarray[K, Wz]): model predicted simcc in z.
    #         simcc_split_ratio (int): The split ratio of simcc.

    #     Returns:
    #         tuple: A tuple containing center and scale.
    #         - np.ndarray[float32]: keypoints in shape (K, 3) or (n, K, 3)
    #         - np.ndarray[float32]: scores in shape (K,) or (n, K)
    #     """
    #     keypoints, scores = self._get_simcc_maximum(simcc_x, simcc_y, simcc_z)
    #     keypoints /= simcc_split_ratio

    #     return keypoints, scores

    def _decode(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                simcc_split_ratio: float=2.0,
                sigma: float=6.0,
                use_dark: bool = False,):
        """Decode SimCC labels into 3D keypoints.

        Args:
            encoded (Tuple[np.ndarray, np.ndarray]): SimCC labels for x-axis,
            y-axis and z-axis in shape (N, K, Wx), (N, K, Wy) and (N, K, Wz)
            simcc_split_ratio (float): The split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - scores (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        keypoints, scores = self._get_simcc_maximum(x, y, z)

        # Unsqueeze the instance dimension for single-instance results
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]
        
        self.sigma = np.array([sigma, sigma, sigma])
        if use_dark:
            x_blur = int((self.sigma[0] * 20 - 7) // 3)
            y_blur = int((self.sigma[1] * 20 - 7) // 3)
            z_blur = int((self.sigma[2] * 20 - 7) // 3)
            x_blur -= int((x_blur % 2) == 0)
            y_blur -= int((y_blur % 2) == 0)
            z_blur -= int((z_blur % 2) == 0)
            keypoints[:, :, 0] = refine_simcc_dark(keypoints[:, :, 0], x,
                                                   x_blur)
            keypoints[:, :, 1] = refine_simcc_dark(keypoints[:, :, 1], y,
                                                   y_blur)
            keypoints[:, :, 2] = refine_simcc_dark(keypoints[:, :, 2], z,
                                                   z_blur)

        keypoints /= simcc_split_ratio
        keypoints_simcc = keypoints.copy()
        keypoints_z = keypoints[..., 2:3]

        keypoints[..., 2:3] = (keypoints_z /
                               (self.input_size[-1] / 2) - 1) * self.z_range
        return keypoints, keypoints_simcc, scores
    

def refine_simcc_dark(keypoints: np.ndarray, simcc: np.ndarray,
                      blur_kernel_size: int) -> np.ndarray:
    """SimCC version. Refine keypoint predictions using distribution aware
    coordinate decoding for UDP. See `UDP`_ for details. The operation is in-
    place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        simcc (np.ndarray): The heatmaps in shape (N, K, Wx)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N = simcc.shape[0]

    # modulate simcc
    simcc = gaussian_blur1d(simcc, blur_kernel_size)
    np.clip(simcc, 1e-3, 50., simcc)
    np.log(simcc, simcc)

    simcc = np.pad(simcc, ((0, 0), (0, 0), (2, 2)), 'edge')

    for n in range(N):
        px = (keypoints[n] + 2.5).astype(np.int64).reshape(-1, 1)  # K, 1

        dx0 = np.take_along_axis(simcc[n], px, axis=1)  # K, 1
        dx1 = np.take_along_axis(simcc[n], px + 1, axis=1)
        dx_1 = np.take_along_axis(simcc[n], px - 1, axis=1)
        dx2 = np.take_along_axis(simcc[n], px + 2, axis=1)
        dx_2 = np.take_along_axis(simcc[n], px - 2, axis=1)

        dx = 0.5 * (dx1 - dx_1)
        dxx = 1e-9 + 0.25 * (dx2 - 2 * dx0 + dx_2)

        offset = dx / dxx
        keypoints[n] -= offset.reshape(-1)

    return keypoints


def gaussian_blur1d(simcc: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate simcc distribution with Gaussian.

    Note:
        - num_keypoints: K
        - simcc length: Wx

    Args:
        simcc (np.ndarray[K, Wx]): model predicted simcc.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the simcc gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, Wx]): Modulated simcc distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    N, K, Wx = simcc.shape

    for n, k in product(range(N), range(K)):
        origin_max = np.max(simcc[n, k])
        dr = np.zeros((1, Wx + 2 * border), dtype=np.float32)
        dr[0, border:-border] = simcc[n, k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, 1), 0)
        simcc[n, k] = dr[0, border:-border].copy()
        simcc[n, k] *= origin_max / np.max(simcc[n, k])
    return simcc