import time
from typing import Any,Tuple,Dict,List,Union,Optional

import cv2
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ...data_classes.segmentation import SAMSegmentationResult
from ...ops.preprocess import standard_preprocess, image_from_path

class SAM2:
    """SAM2 ONNX model for image segmentation.

    Args:
        identifier (str): Model identifier. One of "tiny", "small", "base_plus", "large".
    
    Attributes:
        encoder (SAM2Encoder): The SAM2 encoder model.
        decoder (SAM2Decoder): The SAM2 decoder model.
        orig_im_size (Tuple[int, int]): Original image size (height, width).
        point_coords (Dict[int, np.ndarray]): Dictionary of point coordinates for each label id.
        box_coords (Dict[int, np.ndarray]): Dictionary of box coordinates for each label id.
        point_labels (Dict[int, np.ndarray]): Dictionary of point labels for each label id.
        masks (Dict[int, np.ndarray]): Dictionary of masks for each label id.
    
    """
    def __init__(self, identifier: str) -> None:
        # Initialize models
        self.identifier = identifier
        self.encoder = SAM2Encoder(identifier)
        self.orig_im_size = self.encoder.input_shape[2:]
        self.decoder = None
        
        self.point_coords = {}
        self.box_coords = {}
        self.point_labels = {}
        self.masks = {}

    def set_image(self, image: np.ndarray) -> None:
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
        self.decoder = SAM2Decoder(self.identifier, 
                                   self.encoder.input_shape[2:],
                                   self.orig_im_size)
        self.reset_points()

    def set_point(self, 
                  point_coords: Tuple[int, int], 
                  is_positive: bool, 
                  label_id: int
                  ) -> SAMSegmentationResult:
        '''Set a point to the model and update the mask.

        Args:
            point_coords (Tuple[int, int]): Coordinates of the point (x, y) in (W, H) format.
            is_positive (bool): Whether the point is a positive or negative point.
            label_id (int): Label ID for the point.

        Returns:
            SAMSegmentationResult: Segmentation result with updated masks.
        '''

        if label_id not in self.point_coords:
            self.point_coords[label_id] = np.array([point_coords])
            self.point_labels[label_id] = np.array([1 if is_positive else 0])
        else:
            self.point_coords[label_id] = np.append(self.point_coords[label_id], np.array([point_coords]), axis=0)
            self.point_labels[label_id] = np.append(self.point_labels[label_id], 1 if is_positive else 0)

        return self._decode_mask(label_id)

    def set_box(self, 
                box_coords: Tuple[Tuple[int, int], Tuple[int, int]], 
                label_id: int
                ) -> SAMSegmentationResult:
        '''Set a bounding box for the model and update the mask.

        Args:
            box_coords (Tuple[Tuple[int, int], Tuple[int, int]]): Coordinates of
                the box ((x1, y1), (x2, y2)) in (W, H) format.
            label_id (int): Label ID for the box.
        
        Returns:
            SAMSegmentationResult: Segmentation result with updated masks.
        '''

        point_coords = np.array([box_coords[0], box_coords[1]]) # Convert from 1x4 to 2x2

        self.box_coords[label_id] = point_coords

        return self._decode_mask(label_id)

    def _decode_mask(self, label_id: int
                     ) -> SAMSegmentationResult:
        '''Decode the mask for the given label ID.

        Args:
            label_id (int): Label ID for the mask to decode.

        Returns:
            SAMSegmentationResult: Segmentation result with updated masks.
        '''
        
        concat_coords, concat_labels = self._merge_points_and_boxes(label_id)

        high_res_feats_0, high_res_feats_1, image_embed = self.image_embeddings
        if concat_coords.size == 0:
            mask = np.zeros((self.orig_im_size[0], self.orig_im_size[1]), dtype=np.uint8)
        else:
            mask, _ = self.decoder(image_embed, high_res_feats_0, high_res_feats_1, concat_coords, concat_labels)
        self.masks[label_id] = mask

        return SAMSegmentationResult(
            original_img=self.original_image,
            masks = self.masks)

    def _merge_points_and_boxes(self, label_id: int) -> Tuple[np.ndarray, np.ndarray]:
        concat_coords = []
        concat_labels = []
        has_points = label_id in self.point_coords
        has_boxes = label_id in self.box_coords

        if not has_points and not has_boxes:
            return np.array([]), np.array([])

        if has_points:
            concat_coords.append(self.point_coords[label_id])
            concat_labels.append(self.point_labels[label_id])
        if has_boxes:
            concat_coords.append(self.box_coords[label_id])
            concat_labels.append(np.array([2, 3]))
        concat_coords = np.concatenate(concat_coords, axis=0)
        concat_labels = np.concatenate(concat_labels, axis=0)

        return concat_coords, concat_labels

    def remove_point(self,  
                     point_coords: Tuple[int, int], 
                     label_id: int
                     ) -> SAMSegmentationResult:
        '''Remove a point from the model and update the mask.

        Args:
            point_coords (Tuple[int, int]): Coordinates of the point (x, y) in (W, H) format.
            label_id (int): Label ID for the point.

        Returns:
            SAMSegmentationResult: Segmentation result with updated masks.
        '''
        point_id = np.where((self.point_coords[label_id][:, 0] == point_coords[0]) & (self.point_coords[label_id][:, 1] == point_coords[1]))[0][0]
        self.point_coords[label_id] = np.delete(self.point_coords[label_id], point_id, axis=0)
        self.point_labels[label_id] = np.delete(self.point_labels[label_id], point_id, axis=0)

        return self._decode_mask(label_id)

    def remove_box(self, 
                   label_id: int
                   ) -> SAMSegmentationResult:
        '''Remove a box from the model and update the mask.

        Args:
            label_id (int): Label ID for the box.

        Returns:
            SAMSegmentationResult: Segmentation result with updated masks.
        '''
        del self.box_coords[label_id]
        return self._decode_mask(label_id)

    def get_masks(self) -> Dict[int, np.ndarray]:
        """Get the current masks.
        
        Returns:
            Dict[int, np.ndarray]: Dictionary of masks with label ids as keys.
        """
        return self.masks

    def reset_points(self) -> None:
        """Reset all points, boxes, and masks."""
        self.point_coords = {}
        self.box_coords = {}
        self.point_labels = {}
        self.masks = {}

class SAM2Encoder(Inferencer):
    def __init__(self, identifier: str) -> None:
        self._check_backbone(identifier, 
                    ["tiny", "small","base_plus","large"])
        # Initialize model
        repo_id = f"mabote-itumeleng/ONNX-SAM2-Segment-Anything"
        filename = f"sam2.1_hiera_{identifier}_encoder.onnx"
        self.sess = self._download_and_compile(repo_id, filename)
        self.input_name = self.sess.get_inputs()[0].name

        # Get model info
        self._get_input_details()
        self._get_output_details()

    def __call__(self,image:Union[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = image_from_path(image)
        self.img_height, self.img_width = image.shape[:2]
        input_tensor = self._preprocess(image)

        outputs = self.sess.run(self.output_names, {self.input_names[0]: input_tensor})

        return self._postprocess(outputs)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return standard_preprocess(image, 
                            (self.input_height, self.input_width))

    def _postprocess(self, outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        high_res_feats_0, high_res_feats_1, image_embed = outputs[0], outputs[1], outputs[2]
        return high_res_feats_0, high_res_feats_1, image_embed

    def _get_input_details(self) -> None:
        model_inputs = self.sess.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def _get_output_details(self) -> None:
        model_outputs = self.sess.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


class SAM2Decoder(Inferencer):
    """SAM2 Decoder ONNX model for image segmentation.

    Args:
        identifier (str): Model identifier. One of "tiny", "small", "base_plus", "large".
        encoder_input_size (Tuple[int, int]): Input size of the encoder (height, width).
        orig_im_size (Tuple[int, int], optional): Original image size (height, width). If not provided, it will be set to encoder_input_size.
        mask_threshold (float, optional): Threshold for mask binarization. Default is 0.0.

    """

    def __init__(self, identifier: str,
                 encoder_input_size: Tuple[int, int],
                 orig_im_size: Tuple[int, int] = None,
                 mask_threshold: float = 0.0) -> None:
        self._check_backbone(identifier, 
                    ["tiny", "small","base_plus","large"])
        # Initialize model
        repo_id = f"mabote-itumeleng/ONNX-SAM2-Segment-Anything"
        filename = f"sam2.1_hiera_{identifier}_decoder.onnx"
        self.sess = self._download_and_compile(repo_id, filename)

        self.orig_im_size = orig_im_size if orig_im_size is not None else encoder_input_size
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4

        # Get model info
        self._get_input_details()
        self._get_output_details()

    def __call__(self, image_embed: np.ndarray,
                 high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray,
                 point_coords: np.ndarray, point_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        inputs = self._preprocess(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels)

        outputs = self.sess.run(self.output_names,
                            {self.input_names[i]: inputs[i] for i in range(len(self.input_names))})

        return self._postprocess(outputs)

    def _preprocess(self, image_embed: np.ndarray,
                       high_res_feats_0: np.ndarray, high_res_feats_1: np.ndarray,
                       point_coords: np.ndarray, point_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        input_point_coords, input_point_labels = self._prepare_points(point_coords, point_labels)

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros((num_labels, 1, self.encoder_input_size[0] // self.scale_factor, self.encoder_input_size[1] // self.scale_factor), dtype=np.float32)
        has_mask_input = np.array([0], dtype=np.float32)
        original_size = np.array([self.orig_im_size[0], self.orig_im_size[1]], dtype=np.int32)

        return image_embed, high_res_feats_0, high_res_feats_1, input_point_coords, input_point_labels, mask_input, has_mask_input, original_size


    def _prepare_points(self, point_coords: np.ndarray, point_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        input_point_coords = point_coords[np.newaxis, ...]
        input_point_labels = point_labels[np.newaxis, ...]

        input_point_coords[..., 0] = input_point_coords[..., 0] / self.orig_im_size[1] * self.encoder_input_size[1]  # Normalize x
        input_point_coords[..., 1] = input_point_coords[..., 1] / self.orig_im_size[0] * self.encoder_input_size[0]  # Normalize y

        return input_point_coords.astype(np.float32), input_point_labels.astype(np.float32)

    def _postprocess(self, 
                     outputs: List[np.ndarray]
                     ) -> Tuple[np.ndarray, np.ndarray]:

        scores = outputs[1].squeeze()
        masks = outputs[0]
        masks = masks > self.mask_threshold
        masks = masks.astype(np.uint8).squeeze()

        return masks, scores

    # def _set_image_size(self, orig_im_size: Tuple[int, int]) -> None:
    #     self.orig_im_size = orig_im_size

    def _get_input_details(self) -> None:
        model_inputs = self.sess.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    def _get_output_details(self) -> None:
        model_outputs = self.sess.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]