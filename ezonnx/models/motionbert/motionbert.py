from typing import Any,Tuple,Dict,List,Union,Optional
from copy import deepcopy
import cv2
import numpy as np
import onnxruntime as ort
from ezonnx.core.inferencer import Inferencer
from ..rtmpose.rtmpose import RTMPose
from ...data_classes.object_detection import ObjectDetectionResult,PoseDetectionResult
from ...ops.preprocess import standard_preprocess, image_from_path,coco_h36m
from ...ops.postprocess import apply_savgol_filter_to_skeleton

class MotionBERT3D(Inferencer):
    '''MotionBERT ONNX model for 3D human pose estimation from 2D keypoints.
    
    Args:
        n_frames (int): Number of input frames for the model. Choose from [27,81,243]. Default is 27.
        pose_detector (Inferencer): 2D pose detector. If None, RTMPose "m" model will be used.
        kpt_thresh (float): Keypoint confidence threshold for the 2D pose detector. Default is 0.3.
        onnx_path (Optional[str]): Path to the ONNX model file. If None
    '''
    def __init__(self,
                 n_frames:int=27,
                 pose_detector:Inferencer=None,
                 kpt_thresh = 0.3,
                 onnx_path:Optional[str]=None):
        self.n_frames = n_frames

        if pose_detector is None:
            
            self._pose_det = RTMPose("m",kpt_thresh=kpt_thresh)
        else:
            self._pose_det = pose_detector
        
        # build
        identifier = n_frames
        self._check_backbone(identifier,[27,81,243])
        if onnx_path is None:
            # Initialize model
            repo_id = f"bukuroo/MotionBERT-3d-ONNX"
            filename = f"motionbert_3d_{identifier}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)

        self.kpt_thresh = kpt_thresh

    def __call__(self,
                 video:str=None,
                 poses:np.ndarray=None,
                 stride:int=5,
                 start_frame:int=0,
                 end_frame:int=-1
                 )-> Tuple[np.ndarray,np.ndarray]:
        """Run inference on the input image.

        Args:
            video (str): Input video path.
            poses (np.ndarray): Input 2D poses of shape (N, 17, 2). Either video or poses must be provided.
            stride (int): Number of interval of 3D pose estimation. Default is 5.
            start_frame (int): Start frame for inference. Default is 0.
            end_frame (int): End frame for inference. Default is -1 (till the
        
        Returns:
            Tuple[np.ndarray,np.ndarray]:
                poses_3d (np.ndarray): Output 3D poses of shape (N, 17, 3).
                poses_2d (np.ndarray): Input 2D poses of shape (N, 17, 2).
        """
        if video is None and poses is None:
            raise ValueError("Please provide either video or poses.")
        if video is not None and poses is not None:
            raise ValueError("Please provide either video or poses, not both.")
        if video is not None:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                raise ValueError(f"Error opening video file {video}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            poses_2d = []
            # kpt_scores_all = []
            kpts_old = np.zeros((17,2))
            kpt_score_old = np.zeros((17,))
            # for i in range(frame_count):
            count = 0
            while True:
                ret, frame = cap.read()
                if count<start_frame:
                    count += 1
                    continue
                if end_frame>0 and count>=end_frame:
                    break
                if not ret:
                    break
                det_result:PoseDetectionResult = self._pose_det(frame)

                # normalize keypoints
                if len(det_result.kpts)>0:
                    kpts = normalize_skeleton(det_result.kpts[0])
                    # kpt_scores = det_result.kpt_scores
                    kpts_old = kpts
                    # kpt_score_old = kpt_scores
                else:
                    kpts = kpts_old
                    # kpt_scores = kpt_score_old
                poses_2d.append(kpts)
                # kpt_scores_all.append(kpt_scores)
                count += 1
                print(f"\rExtracting 2d poses {count}/{frame_count}", end="\r")
            cap.release()
        else:
            poses_2d = poses
        # convert h36m format
        poses_2d,_ = coco_h36m(np.array(poses_2d))
        # apply filter
        poses_2d = apply_savgol_filter_to_skeleton(poses_2d,window_length=25)
        # convert to 3d
        poses_3d = pose2d_to_3d(self.sess,
                                poses_2d,
                                input_len=self.n_frames,
                                stride=stride
                                )
        # align spine to y axis
        poses_3d = align_skeleton_to_y_axis(poses_3d)

        return poses_3d, poses_2d

    def _preprocess(self,image:np.ndarray)-> np.ndarray:
        pass
    def _postprocess(self, output: List[np.ndarray]) -> List:
        pass

# y軸中心での向きを揃える関数
def align_skeleton_y_rotation(data):
    """
    4->1,11->14の平均ベクトルがx軸方向に並行になるように補正する

    Parameters:
    data (numpy.ndarray): Input array of shape (N, 17, 3) containing 3D skeleton coordinates.

    Returns:
    numpy.ndarray: Rotated array of shape (N, 17, 3).
    """
    # Compute vectors over all frames
    v1 = np.mean(data[:, 1, :] - data[:, 4, :], axis=0)  # Vector from point 4 to 1
    v2 = np.mean(data[:, 14, :] - data[:, 11, :], axis=0)  # Vector from point 11 to 14

    # Compute the average vector
    v = (v1 + v2) / 2

    # Extract xz components
    v_xz = np.array([v[0], v[2]])

    # Compute rotation angle θ
    theta = -np.arctan2(v_xz[1], v_xz[0])

    # Construct the rotation matrix A around the y-axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    A = np.array([
        [cos_theta,  0, sin_theta],
        [0,          1,         0],
        [-sin_theta, 0, cos_theta]
    ])  # Shape: (3, 3)

    # Apply rotation matrix A to all data points
    data_rotated = data @ A  # Shape: (N, 17, 3)

    return data_rotated

# y軸と背骨を並行にする関数
def align_skeleton_to_y_axis(skeleton_data):
    """
    (フレーム数, 17, 3)の3D骨格座標データを、ポイント10からポイント0への平均ベクトルが
    Y軸に並行になるように回転させる関数。

    Parameters:
    - skeleton_data: np.ndarray
        形状 (フレーム数, 17, 3) の3D骨格座標データ

    Returns:
    - rotated_skeleton_data: np.ndarray
        指定の回転を適用した (フレーム数, 17, 3) の3D骨格座標データ
    """
    num_frames, num_joints, num_coords = skeleton_data.shape
    assert num_joints >= 11 and num_coords == 3, "入力データは少なくとも11個の関節と3次元座標が必要です。"

    # 各フレームでポイント10からポイント0へのベクトルを計算
    vectors_10_to_0 = skeleton_data[:, 0, :] - skeleton_data[:, 10, :]

    # ベクトルの平均を計算
    mean_vector = np.mean(vectors_10_to_0, axis=0)

    # 平均ベクトルを正規化
    mean_vector_norm = mean_vector / np.linalg.norm(mean_vector)

    # Y軸方向の単位ベクトル
    y_axis = np.array([0, 1, 0])

    # 回転軸を計算（平均ベクトルとY軸の外積）
    rotation_axis = np.cross(mean_vector_norm, y_axis)

    # 外積のノルムが0の場合（既にY軸に平行な場合）、回転不要
    if np.linalg.norm(rotation_axis) == 0:
        rotated_skeleton_data = skeleton_data.copy()
        return rotated_skeleton_data

    # 回転角度を計算
    angle = np.arccos(np.clip(np.dot(mean_vector_norm, y_axis), -1.0, 1.0))

    # 回転軸を正規化
    rotation_axis_norm = rotation_axis / np.linalg.norm(rotation_axis)

    # Rodriguesの回転公式を使用して回転行列を計算
    K = np.array([
        [0, -rotation_axis_norm[2], rotation_axis_norm[1]],
        [rotation_axis_norm[2], 0, -rotation_axis_norm[0]],
        [-rotation_axis_norm[1], rotation_axis_norm[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # 全フレームに回転行列を適用
    rotated_skeleton_data = np.empty_like(skeleton_data)
    for i in range(num_frames):
        rotated_skeleton_data[i] = np.dot(skeleton_data[i], R.T)

    return rotated_skeleton_data

def pose2d_to_3d(session:ort.InferenceSession,
                poses_2d:np.ndarray,
                # kpt_scores_all:np.ndarray,
                input_len:int=27,
                stride:int=5
                )-> np.ndarray:
    """Convert 2D poses to 3D poses using the MotionBERT model.

    Args:
        session (ort.InferenceSession): MotionBERT session.
        poses_2d (np.ndarray): Input 2D poses of shape (N, 17, 2).
        kpt_scores_all (np.ndarray): Input keypoint confidence scores of shape (N, 17).
    
    Returns:
        np.ndarray: Output 3D poses of shape (N, 17, 3).
    """
    start_id=0
    len_sample = len(poses_2d)
    stacked_pose_3d = []
    while True:
        try:
            padding_length=0
            # 左右反転の水増しをして平均を取る
            # 1 input
            input_2D=poses_2d[start_id:start_id+input_len].astype("float32")
            # score = kpt_scores_all[start_id:start_id+input_len].astype("float32")
            if len(input_2D)<input_len:
                padding_length = input_len - len(input_2D)
                last_pose = input_2D[-1]  # 最後の姿勢
                # 最後の姿勢をpadding_length分だけ繰り返す
                padding = np.tile(last_pose, (padding_length, 1, 1))
                # パディングを元の配列に結合
                input_2D = np.concatenate([input_2D, padding], axis=0)
            joints_left =  [4, 5, 6, 11, 12, 13]
            joints_right = [1, 2, 3, 14, 15, 16]
            input_2D_aug = deepcopy(input_2D)
            input_2D_aug[ :, :, 0] *= -1
            input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
            # motionbertの入力は(1,27,17,3) 3は[x,y,conf]
            # infer
            input_2D = np.concatenate([input_2D, np.ones((len(input_2D), 17, 1))], axis=-1).astype("float32")
            input= np.expand_dims(input_2D,axis=0)
            out = session.run(None,{"input":input})[0][0]
            
            # infer with augmented data
            input_2D_aug = np.concatenate([input_2D_aug, np.ones((len(input_2D_aug), 17, 1))], axis=-1).astype("float32")
            input_aug= np.expand_dims(input_2D_aug,axis=0)
            out_aug = session.run(None,{"input":input_aug})[0][0]

            out_aug[ :, :, 0] *= -1
            out_aug[ :, joints_left + joints_right, :] = out_aug[ :, joints_right + joints_left, :] 

            mean_out = (out + out_aug) / 2
            stacked_pose_3d.append(mean_out.tolist())
            start_id+=stride
            print(f"\rConverting to 3D poses {start_id}/{len_sample}", end="\r")
            if padding_length>0:
                break
        except Exception as e:
            print(e)
            break
    stacked_pose_3d = np.array(stacked_pose_3d)
    poses_3d = average_overlapping_skeletons(stacked_pose_3d,stride)
    if padding_length>0:
        poses_3d = poses_3d[:-padding_length]
    return poses_3d

def normalize_skeleton(data):
    """
    COCO形式の17点の骨格データを正規化し、オフセットする関数。

    Parameters:
    - data: shape (17, 2) のnumpy配列。各行が[x, y]座標を表す。

    Returns:
    - 正規化およびオフセットされた同じ形状のnumpy配列。
    """
    data = data.copy()  # 元のデータを変更しないようにコピー

    # 正規化に使用するペアのリスト
    pairs = [
        (5, 11),  # 左肩と左腰
        (6, 12),  # 右肩と右腰
        (11, 13), # 左腰と左ひざ
        (12, 14)  # 右腰と右ひざ
    ]

    scale_factor = None

    for idx1, idx2 in pairs:
        point1 = data[idx1]
        point2 = data[idx2]

        # xとyがfloat型かどうかを確認
        if (np.issubdtype(point1[0].dtype, np.floating) and np.issubdtype(point1[1].dtype, np.floating) and
            np.issubdtype(point2[0].dtype, np.floating) and np.issubdtype(point2[1].dtype, np.floating)):
            
            # 欠損値（NaN）がないか確認
            if not (np.isnan(point1).any() or np.isnan(point2).any()):
                # 距離を計算
                distance = np.linalg.norm(point1 - point2)
                if distance != 0:
                    scale_factor = distance * 2
                    break

    if scale_factor is None:
        raise ValueError("有効なペアが見つからず、正規化できません。")

    # データを正規化
    data /= scale_factor

    # 左腰と右腰の中点を計算
    hips = []
    for idx in [11, 12]:
        point = data[idx]
        if (np.issubdtype(point[0].dtype, np.floating) and np.issubdtype(point[1].dtype, np.floating)):
            if not np.isnan(point).any():
                hips.append(point)
    
    if hips:
        hips_midpoint = np.mean(hips, axis=0)
    else:
        hips_midpoint = np.array([0.0, 0.0])

    # オフセットを適用
    data -= hips_midpoint

    return data

def average_overlapping_skeletons(skeleton_data, stride):
    """
    (K, 27, 17, 3)の3D骨格座標データを、重複するフレームを平均して
    (フレーム数, 17, 3)に変換する関数。ウィンドウを進めるフレーム数を
    引数で指定できます。

    Parameters:
    - skeleton_data: np.ndarray
        形状 (K, window_size, 17, 3) の入力データ
        K: ウィンドウ数（window_sizeフレームごとにstrideフレーム進めたデータ）
        window_size: 各ウィンドウのフレーム数（この場合27）
        17: 関節数
        3: 3次元座標 (x, y, z)
    - stride: int
        ウィンドウを進めるフレーム数（kフレーム）

    Returns:
    - averaged_skeletons: np.ndarray
        形状 (フレーム数, 17, 3) の平均化された3D骨格データ
    """
    K, window_size, num_joints, num_coords = skeleton_data.shape

    # 全体のフレーム数を計算
    total_frames = (K - 1) * stride + window_size

    # 各フレームのデータを保存するための配列 (フレーム数, 17, 3)
    summed_skeletons = np.zeros((total_frames, num_joints, num_coords))
    frame_counts = np.zeros(total_frames)

    # 各ウィンドウ (K個) のデータをフレームごとに加算
    for i in range(K):
        start_frame = i * stride
        end_frame = start_frame + window_size
        frames = slice(start_frame, end_frame)

        summed_skeletons[frames] += skeleton_data[i]
        frame_counts[frames] += 1

    # 加算したデータをカウントで割り、平均化
    averaged_skeletons = summed_skeletons / frame_counts[:, np.newaxis, np.newaxis]

    return averaged_skeletons