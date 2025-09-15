from typing import Any,Tuple,Dict,List,Union,Optional
import numpy as np
from ezonnx.core.inferencer import Inferencer
from ezonnx.data_classes.classification import ClassificationResult
from ezonnx.ops.preprocess import standard_preprocess, image_from_path

class ImageClassifier(Inferencer):
    """ONNX inferencer for the image classification models. 
    The model input must be (N,3,H,W) and output must be (N,num_classes).
    ConvNeXT, EfficientNetV2, MobileNetV4, CoAtNet, ViT etc.

    Args:
        onnx_path (str): Path to a local ONNX model file. Model input must be (N,3,H,W) and output must be (N,num_classes).
        size (int, optional): Size to resize the input image. If None, use the model's default input size. Defaults to None.
        standardize (bool, optional): Whether to standardize the input image.(Center value to 0) Defaults to True.
        center_crop (bool, optional): Whether to apply center cropping to the input image. Defaults to True.
        std (List[float], optional): Standard deviation for each channel. Defaults to [0.485, 0.456, 0.406].
        mean (List[float], optional): Mean for each channel. Defaults to [0.229, 0.224, 0.225].
    
    Examples:
        Usage example:
        ::
            from ezonnx import ImageClassifier

            model = ImageClassifier("path/to/your_model.onnx")  # ConvNeXT, EfficientNet, MobileNet, CoatNet, ViT etc.
            result = model("image.jpg")

            print(result.class)  # (1,) predicted class index
            print(result.score)  # (1,) predicted class score
            print(result.logits)  # (N,) predicted class logits
            print(result.original_img)  # (H,W,3) original image
    """

    def __init__(self,
                 onnx_path:str,
                 size:Optional[int]=None,
                 standardize:bool=True,
                 center_crop:bool=True,
                 std:Tuple[float,float,float]=(0.485, 0.456, 0.406),
                 mean:Tuple[float,float,float]=(0.229, 0.224, 0.225)) -> None:
        self.standardize = standardize
        self.center_crop = center_crop
        self.std = std
        self.mean = mean

        self.sess = self._compile_from_path(onnx_path)

        if size is not None:
            self.input_size = (size,size)
        else:
            self.input_size = self.sess.get_inputs()[0].shape[2:]
            if isinstance(self.input_size[0], str):
                raise ValueError("Input size is dynamic. Please specify 'size' argument to resize")
        self.input_name = self.sess.get_inputs()[0].name
        

    def __call__(self,image:Union[str, np.ndarray])-> ClassificationResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
        
        Returns:
            ClassificationResult: Inference result containing class index, score, logits, and original image.
        """
        image = image_from_path(image)
        input_tensor = self._preprocess(image)
        outputs = self.sess.run(None, {self.input_name: input_tensor})
        class_id, score,logits = self._postprocess(outputs)

        return ClassificationResult(
            original_img = image,
            class_id = class_id,
            score = score,
            logits = logits
        )

    def _preprocess(self,image:np.ndarray)-> np.ndarray:
        """Preprocess the input image for model inference.

        Args:
            image (np.ndarray): Input image array.
        
        Returns:
            np.ndarray: Preprocessed image tensor ready for model input.
        """
        if self.center_crop:
            h, w = image.shape[:2]
            crop_size = min(h, w)
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
            image = image[top:top+crop_size, left:left+crop_size]
        input_tensor = standard_preprocess(image,
                                           self.input_size,
                                           standardize=self.standardize,
                                           mean=self.mean,
                                           std=self.std)
        return input_tensor
    
    def _postprocess(self,outputs:List[np.ndarray])-> Tuple[int,float,np.ndarray]:
        """Postprocess the model outputs to extract class index, score, and logits.

        Args:
            outputs (List[np.ndarray]): Raw model outputs.
        
        Returns:
            Tuple[int,float,np.ndarray]: Predicted class index, score, and logits.
        """
        logits = outputs[0][0]
        class_id = int(np.argmax(logits))
        # apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        score = float(probs[class_id])
        return class_id, score, logits
