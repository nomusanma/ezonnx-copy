from typing import Tuple, Union
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ...ops.preprocess import image_from_path
from ...data_classes.keypoint_detection import KeypointDetectionResult
from ...data_classes.image_matching import ImageMatchingResult
from ...core.inferencer import Inferencer
from ...models.lightglue.lightglue import LightGlue

class ALIKEDLightGlue:
    def __init__(self, identifier:str,size:int=640):
        """
        Args:
            aliked_path: str path to the aliked onnx model
            lightglue_path: str path to the lightglue onnx model
            score_thresh: float score threshold for filtering matches
        """
        self.aliked = ALIKED(identifier,size)
        self.lightglue = LightGlue("aliked")

    def __call__(self,image0,image1)->ImageMatchingResult:
        """
        Args:
            image0: str|np.ndarray shape (H,W,3) BGR, if str, it should be the name of the template
            image1: np.ndarray shape (H,W,3) BGR
        Returns:
            m_kpts0: np.ndarray shape (K,2) matched coordinates of keypoints
            m_kpts1: np.ndarray shape (K,2) matched coordinates of keypoints
            scores: np.ndarray shape (K,) scores of the matches
        """
        image0 = image_from_path(image0)
        image1 = image_from_path(image1)
        self.ret0= self.aliked(image0)
        self.ret1= self.aliked(image1)
        matches,scores = self.lightglue(self.ret0.kpts_norm, 
                                        self.ret1.kpts_norm, 
                                        self.ret0.descs,
                                        self.ret1.descs)
        return ImageMatchingResult(
            original_img=image0,
            query_img=image1,
            m_kpts0=self.ret0.kpts[matches[:,0]],
            m_kpts1=self.ret1.kpts[matches[:,1]],
            scores=scores)
        
    
    def register_template(self, name:str,image:np.ndarray)->None:
        """
        register a template image to be used in the future
        Args:
            name: str template name
            image: np.ndarray shape (H,W,3) BGR
        """
        scale,tensor = self.preprocess(image)
        kpt,desc,score = self.aliked.run(None, {self.input_name: tensor})
        self.templates[name]=(kpt,desc,score,scale)
    
    def padding(self, image:np.ndarray)->Tuple[float,np.ndarray]:
        """
        resize the image keeping the aspect ratio
        and pad it to the input shape (letterbox process)
        Args:
            image: np.ndarray shape (H,W,3) BGR
        Returns:
            scale: float resize scale
            padded: np.ndarray shape (H',W',3) BGR
        """
        size=self.input_shape
        h, w = image.shape[:2]
        resize_scale = min(size[0] /h, size[1] / w)
        scale = 1 / max(h, w)
        h_new, w_new = int(h * resize_scale), int(w * resize_scale)
        image = cv2.resize(image, (w_new, h_new))

        padded = np.zeros((size[0], size[1], 3), dtype=np.float32)
        padded[:h_new, :w_new] = image

        return scale,padded.astype("uint8")

    def preprocess(self, image)->Tuple[float,np.ndarray]:
        """
        process the image to the format that the model accepts
        Args:
            image: np.ndarray shape (H,W,3) BGR
        Returns:
            scale: float resize scale
            tensor: np.ndarray shape (1,3,H,W) RGB
        """
        # resize the image and pad it
        scale,padded_image = self.padding(image)
        # convert to RGB
        padded_image_rgb = padded_image[:, :, ::-1]
        # normalize the image
        tensor = padded_image_rgb / 255.0
        # HWC -> CHW
        tensor = tensor.transpose(2, 0, 1)
        # add batch dimension
        tensor = np.expand_dims(tensor, axis=0)
        return scale,tensor.astype(np.float32)
    
    def find_matches_from_kpts(self,kpts0,kpts1,desc0,desc1,scales0, scales1):
        """
        find matches between two sets of keypoints
        Args:
            kpts0: np.ndarray shape (N,2) normalized coordinates of keypoints
            kpts1: np.ndarray shape (M,2) normalized coordinates of keypoints
            desc0: np.ndarray shape (N,D)
            desc1: np.ndarray shape (M,D)
            scales0: float
            scales1: float
        Returns:
            m_kpts0: np.ndarray shape (K,2) matched coordinates of keypoints
            m_kpts1: np.ndarray shape (K,2) matched coordinates of keypoints
        """
        # find matches
        matches0, mscores0 = self.lightglue.run(
            None,
            {
                "kpts0": kpts0[None,:], # add batch dimension
                "kpts1": kpts1[None,:],
                "desc0": desc0[None,:],
                "desc1": desc1[None,:],
            },
        )
        # filter matches by score
        matches0 = matches0[mscores0>self.score_thresh]
        # postprocess the matches
        m_kpts0, m_kpts1 = self.postprocess(
            kpts0, kpts1, matches0, scales0, scales1
        )
        scores = mscores0[mscores0>self.score_thresh]
        return m_kpts0, m_kpts1, scores
    
    @staticmethod
    def postprocess(kpts0, kpts1, matches, scales0, scales1):
        """
        postprocess the matches
        Args:
            kpts0: np.ndarray shape (N,2) normalized coordinates of keypoints
            kpts1: np.ndarray shape (M,2) normalized coordinates of keypoints
            matches: np.ndarray shape (K,2) indices of matches
            scales0: float
            scales1: float
        Returns:
            m_kpts0: np.ndarray shape (K,2) matched coordinates of keypoints
            m_kpts1: np.ndarray shape (K,2) matched coordinates of keypoints
        """
        # denormalize the keypoints
        kpts0 = (kpts0 + 1) / scales0 / 2 
        kpts1 = (kpts1 + 1) / scales1 / 2
        # create match indices
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        return m_kpts0, m_kpts1
    
    @staticmethod
    def transform_image(ref_img,target_img,ref_points,target_points):
        """
        affine transformation of the target image to the reference image
        Args:
            ref_img: np.ndarray shape (H,W,3) BGR
            target_img:  np.ndarray shape (H,W,3) BGR
            ref_points: matched keypoints in reference image, shape (N,2)
            target_points: matched keypoints in target image, shape (N,2)
        Returns:
            success: bool
            transformed_image: np.ndarray shape (H,W,3) BGR
        """
        
        ref_points = np.array(ref_points, dtype=np.float32)
        target_points = np.array(target_points, dtype=np.float32)
        
        # calculate homography
        # M, _ = cv2.estimateAffinePartial2D(ref_points,target_points, method=cv2.RANSAC)
        M, mask = cv2.findHomography(ref_points, target_points, cv2.RANSAC, 5.0)
        
        if M is not None:
            # affine transformation
            # transformed_image = cv2.warpAffine(target_img, M, (ref_img.shape[1], ref_img.shape[0]))
            transformed_image = cv2.warpPerspective(target_img, np.linalg.inv(M),  (ref_img.shape[1], ref_img.shape[0]))
            return True,transformed_image
        else:
            print("Failed to find homography")
            return False,target_img
        
    @staticmethod    
    def draw_matches(ref_img,target_img,ref_points,target_points,scores):
        """
        draw the matched pairs in image
        Args:
            ref_img: np.ndarray shape (H,W,3) BGR
            target_img:  np.ndarray shape (H,W,3) BGR
            ref_points: matched keypoints in reference image, shape (N,2)
            target_points: matched keypoints in target image, shape (N,2)
            scores: np.ndarray shape (N,) scores of the matches
        Returns:
            None
        """
        marged_width = ref_img.shape[1] + target_img.shape[1]
        marged_height = max(ref_img.shape[0], target_img.shape[0])
        matches_img = np.zeros((marged_height, marged_width, 3), dtype=np.uint8)
        matches_img[:ref_img.shape[0], :ref_img.shape[1]] = ref_img
        matches_img[:target_img.shape[0], ref_img.shape[1]:] = target_img
        for ref_point, target_point,score in zip(ref_points, target_points, scores):
            ref_point = ref_point.astype(int)
            target_point = target_point.astype(int)
            target_point[0] += ref_img.shape[1]
            color = (0, int(255 * score), int(255 * (1 - score)))
            cv2.line(matches_img, tuple(ref_point), tuple(target_point),color, 1)
        return matches_img
    
    def show_result(self,img0,img1,m_kpts0,m_kpts1,score):
        """
        visualize the matches and transformed image
        Args:
            img0: np.ndarray shape (H,W,3) BGR
            img1:  np.ndarray shape (H,W,3) BGR
            m_kpts0: matched keypoints in image0, shape (N,2)
            m_kpts1: matched keypoints in image1, shape (N,2)
            score: np.ndarray shape (N,) scores of the matches
        Returns:
            None
        """
        ret,transformed_img = self.transform_image(img0,img1,m_kpts0,m_kpts1)
        matches_img = self.draw_matches(img0,img1,m_kpts0,m_kpts1,score)

        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[matches_img.shape[1]/matches_img.shape[0], 
                                                transformed_img.shape[1]/transformed_img.shape[0]])
        a0 = fig.add_subplot(gs[0])
        a1 = fig.add_subplot(gs[1])
        
        a0.imshow(matches_img[..., ::-1])
        a0.set_title("Matches")
        a0.axis("off")
        a1.imshow(transformed_img[..., ::-1])
        a1.set_title("Transformed image")
        a1.axis("off")
        plt.show()


class ALIKED(Inferencer):
    def __init__(self, identifier=None, threshold: float = 0.2, size=640, model_path=None):
        # モデルパスが指定されていればローカルファイルを使う
        if model_path is not None:
            self.sess = ort.InferenceSession(model_path)
        else:
            self._check_backbone(identifier, ["16rot-top1k", "16rot-top2k", "32-top2k"])
            repo_id = "bfukuroo/ALIKED-LightGlue-ONNX"
            filename = f"aliked-{identifier}-{size}.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape[2:]
        self.threshold = threshold

    def __call__(self,image:Union[str, np.ndarray])-> KeypointDetectionResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            KeypointDetectionResult: Inference result containing keypoints, descriptors and scores.
        """
        image = image_from_path(image)
        scale,tensor = self._preprocess(image)
        # keypoits are in normalized coordinates (-1 to 1)
        kpts_norm,descs,scores = self.sess.run(None, {self.input_name: tensor})
        # denormalize and filter by score
        kpts = self._postprocess(kpts_norm,scale)
        return KeypointDetectionResult(
            original_img=image,
            kpts=kpts,
            kpts_norm=kpts_norm,
            descs=descs,
            scores=scores
        )   
    def _postprocess(self, 
                     kpts:np.ndarray, 
                     scale:float
                     )->np.ndarray:
        """Postprocess the model outputs.
        
        Args:
            kpts (np.ndarray): Keypoints from the model, shape (N, 2) normalized coordinates.
            scale (float): Scale factor used during preprocessing.
        Returns:
            np.ndarray: Denormalized keypoints in shape (N, 2).
        """
        kpts = (kpts + 1) / scale / 2  # denormalize the keypoints
        return kpts
    

    def _padding(self, image:np.ndarray)->Tuple[float,np.ndarray]:
        """
        resize the image keeping the aspect ratio
        and pad it to the input shape (letterbox process)
        Args:
            image: np.ndarray shape (H,W,3) BGR
        Returns:
            scale: float resize scale
            padded: np.ndarray shape (H',W',3) BGR
        """
        size=self.input_shape
        h, w = image.shape[:2]
        resize_scale = min(size[0] /h, size[1] / w)
        scale = 1 / max(h, w)
        h_new, w_new = int(h * resize_scale), int(w * resize_scale)
        image = cv2.resize(image, (w_new, h_new))

        padded = np.zeros((size[0], size[1], 3), dtype=np.float32)
        padded[:h_new, :w_new] = image

        return scale,padded.astype("uint8")

    def _preprocess(self, image)->Tuple[float,np.ndarray]:
        """
        process the image to the format that the model accepts
        Args:
            image: np.ndarray shape (H,W,3) BGR
        Returns:
            scale: float resize scale
            tensor: np.ndarray shape (1,3,H,W) RGB
        """
        # resize the image and pad it
        scale,padded_image = self._padding(image)
        # convert to RGB
        padded_image_rgb = padded_image[:, :, ::-1]
        # normalize the image
        tensor = padded_image_rgb / 255.0
        # HWC -> CHW
        tensor = tensor.transpose(2, 0, 1)
        # add batch dimension
        tensor = np.expand_dims(tensor, axis=0)
        return scale,tensor.astype(np.float32)

