from typing import Optional, Tuple, Union, List
import numpy as np
import cv2
from ..sam2.sam2 import SAM2, SAM2Encoder, SAM2Decoder
from ...ops.preprocess import image_from_path

class EdgeTAM(SAM2):
    def __init__(self,mask_thresh=-0.3) -> None:
        # Initialize models
        self.mask_thresh = mask_thresh
        self.encoder = EdgeTAMEncoder()
        self.orig_im_size = self.encoder.input_shape[2:]
        self.decoder = None
        
        
        self.point_coords = {}
        self.box_coords = {}
        self.point_labels = {}
        self.masks = {}
        self.scores = {}

    def set_image(self, image: Union[str,np.ndarray]) -> None:
        '''Set the input image and extract image embeddings.
        
        Args:
            image (np.ndarray): Input image in shape (H, W, 3).

        Returns:
            None
        '''
        image = image_from_path(image)
        self.original_image = image
        self.image_embeddings = self.encoder(image)
        self.orig_im_size = (image.shape[0], image.shape[1])
        if self.decoder is not None:
            del self.decoder # free memory
        self.decoder = EdgeTAMDecoder(self.encoder.input_shape[2:],
                                   self.orig_im_size,
                                   mask_threshold=self.mask_thresh) # diff point from SAM2
        self.reset_points()

class EdgeTAMEncoder(SAM2Encoder):
    """EdgeTAM Encoder ONNX model for image segmentation.

    Args:
        onnx_path (str, optional): Path to the ONNX model file. If not provided, the model will be downloaded.
    """
    def __init__(self, 
                 onnx_path: Optional[str] = None) -> None:
        if onnx_path is None:
            # Initialize model
            repo_id = f"bukuroo/EdgeTAM-ONNX"
            filename = f"edgetam_encoder.onnx"
            self.sess = self._download_and_compile(repo_id, filename)
        else:
            self.sess = self._compile_from_path(onnx_path)

        self.input_name = self.sess.get_inputs()[0].name

        # Get model info
        self._get_input_details()
        self._get_output_details()

    def _postprocess(self, outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_embed, high_res_feats_0, high_res_feats_1 = outputs[0], outputs[1], outputs[2]
        return high_res_feats_0, high_res_feats_1, image_embed

class EdgeTAMDecoder(SAM2Decoder):
    """EdgeTAM Decoder ONNX model for image segmentation.

    Args:
        encoder_input_size (Tuple[int, int]): Input size of the encoder (height, width).
        orig_im_size (Tuple[int, int], optional): Original image size (height, width). If not provided, it will be set to encoder_input_size.
        mask_threshold (float, optional): Threshold for mask binarization. Default is 0.0.
        onnx_path (str, optional): Path to the ONNX model file. If not provided, the model will be downloaded.
    """

    def __init__(self, 
                 encoder_input_size: Tuple[int, int],
                 orig_im_size: Optional[Tuple[int, int]] = None,
                 mask_threshold: float = 0.0,
                 onnx_path: Optional[str] = None) -> None:
        # Initialize model
        repo_id = f"bukuroo/EdgeTAM-ONNX"
        filename = f"edgetam_decoder.onnx"
        self.sess = self._download_and_compile(repo_id, filename)

        self.orig_im_size = orig_im_size if orig_im_size is not None else encoder_input_size
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4

        # Get model info
        self._get_input_details()
        self._get_output_details()

    
    def _preprocess(self, image_embed: np.ndarray,
                       high_res_feats_0: np.ndarray, 
                       high_res_feats_1: np.ndarray,
                       point_coords: np.ndarray, 
                       point_labels: np.ndarray
                       )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        input_point_coords, input_point_labels = self._prepare_points(point_coords, point_labels)

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros((num_labels, 1, self.encoder_input_size[0] // self.scale_factor, self.encoder_input_size[1] // self.scale_factor), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        # original_size = np.array([self.orig_im_size[0], self.orig_im_size[1]], dtype=np.int32)

        return image_embed, high_res_feats_0, high_res_feats_1, input_point_coords, input_point_labels, mask_input, has_mask_input

    def _postprocess(self, 
                     outputs: List[np.ndarray]
                     ) -> Tuple[np.ndarray, float]:

        scores = outputs[1].squeeze()
        score = float(np.max(scores))
        masks = outputs[0]
        
        masks = np.transpose(masks, (0,2,3,1))
        mask = masks[...,np.argmax(scores)][0]
        mask = cv2.resize(mask, (self.orig_im_size[1],self.orig_im_size[0]), interpolation=cv2.INTER_LINEAR)

        mask = mask > self.mask_threshold
        mask = mask.astype(np.uint8).squeeze()

        return mask, float(score)