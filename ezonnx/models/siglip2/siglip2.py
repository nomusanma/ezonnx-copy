from typing import Optional, Union, List
import cv2
import numpy as np
from ...core.inferencer import Inferencer
from ...ops.preprocess import standard_preprocess, image_from_path
from ...data_classes.zeroshot_classification import ZeroshotClassificationResult
from transformers import AutoTokenizer

class Siglip2(Inferencer):
    """Siglip2 ONNX model for image and text feature extraction.
    Including probability for zero-shot image classification.

    Args:
        name (str): Model name, e.g., "base-patch16-256". Default is "base-patch16-256".
        quantize (Optional[str]): Quantization type, e.g., "q4", "quantized". Default is None.
    
    Examples:
        ::

            from ezonnx import Siglip2
            siglip2 = Siglip2()
            result = siglip2(["a cat", "a dog"], "image.jpg")
            print(result["image_patch_tokens"].shape)  # (N, D) D depends on the model
            print(result["image_features"].shape)  # (H, W, 3)
            print(result["text_token"].shape)  # (N, 1)
            print(result["text_features"].shape)  # (N, D)
            print(result["prob"])  # Probability for each text label
    """


    def __init__(self,
                 name: str = "base-patch16-256",
                 quantize: Optional[str] = None):
        repo_id = f"onnx-community/siglip2-{name}-ONNX"
        text_model_filename = "onnx/text_model.onnx"
        vision_model_filename = "onnx/vision_model.onnx"
        self.text_sess = self._download_and_compile(repo_id, text_model_filename, quantize)
        self.vision_sess = self._download_and_compile(repo_id, vision_model_filename, quantize)
                # モデルの入力名を取得
        self.vision_input_name = self.vision_sess.get_inputs()[0].name
        self.text_input_ids_name = self.text_sess.get_inputs()[0].name
        self.size = int(name.split("-")[-1])
        self.patch = int(name.split("-")[-2].replace("patch", ""))

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    def __call__(self,image:Union[str, np.ndarray],texts:Union[List[str],str])-> ZeroshotClassificationResult:
        """Run inference on the input image.

        Args:
            image (Union[str, np.ndarray]): Input image path or image array.
            texts (Union[List[str], str]): Input text or list of texts.

        Returns:
            FeatureExtractionResult: Inference result containing class and patch tokens.
        """

        return self._infer(image,texts)
    
    
    def _preprocess(self, image:np.ndarray):
        """Preprocess the input image for the model.

        Args:
            image (np.ndarray): Input image array. 
        
        Returns:
            np.ndarray: Preprocessed image tensor in shape (1, 3, H, W).
        """

        return standard_preprocess(image, 
                                   (self.size, self.size),
                                    std=(0.5, 0.5, 0.5),
                                    mean=(0.5, 0.5, 0.5))
    
    def _preprocess_text(self,texts:Union[List[str],str]):
        """Preprocess the input texts for the model.

        Args:
            texts (Union[List[str], str]): Input text or list of texts.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed input IDs and attention mask.
        """
        if isinstance(texts, str):
            texts = [texts]
        # IMPORTANT: pass `padding=max_length` and `max_length=64` since the model was trained with this
        text_inputs = self.tokenizer(texts, 
                        padding="max_length", 
                        max_length=64,
                        return_attention_mask=True,  # attention_mask is required for integrated model
                        return_tensors="np")
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        return input_ids,attention_mask
    
    def _infer(self,
               image:Union[str, np.ndarray], 
               texts:Union[List[str],str]) -> ZeroshotClassificationResult:
        
        image = image_from_path(image)
        input_ids,attention_mask = self._preprocess_text(texts)
        image_tensor = self._preprocess(image)


        # 前処理した画像をモデルに入力して特徴ベクトルを取得
        patch_tokens,im_feature = self.vision_sess.run(None, 
                                                        {self.vision_input_name: image_tensor})
        # テキストをモデルに入力して特徴ベクトルを取得
        text_tokens,text_features= self.text_sess.run(None, 
                                                     {self.text_input_ids_name: input_ids})
        
        prob = self._postprocess(patch_tokens,im_feature,text_tokens, text_features)
        return ZeroshotClassificationResult(
            original_img=image,
            size=self.size,
            patch=self.patch,
            patch_tokens=patch_tokens[0],  # (N, D)
            class_token=im_feature[0],  # (D,)
            prob=prob,
            texts=texts,
            text_tokens=text_tokens,  # (N, 64, D)
            text_features=text_features  # (N, D)
        )

    
    def _postprocess(self,patch_tokens:np.ndarray,im_feature:List[np.ndarray],
                     text_tokens:np.ndarray, text_features:List[np.ndarray]) -> dict:
        """Postprocess the model outputs to extract patch tokens and class tokens."""

        similarities = self._cal_similarity(im_feature[0],text_features)
        prob_pre = self._sigmoid(similarities)
        prob = prob_pre/sum(prob_pre)*100
        return prob

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def _cal_similarity(self,image_feature,txt_features):
        # コサイン類似度の計算
        # ベクトルを正規化
        # image_vec = image_features[0]  # 1xD -> D
        image_feature_norm = image_feature / np.linalg.norm(image_feature)
        # text_vecs = text_features     # NxD
        txt_features_norm = txt_features / np.linalg.norm(txt_features, axis=1, keepdims=True)

        # 画像ベクトルと各テキストベクトルとのコサイン類似度を計算
        similarities = txt_features_norm.dot(image_feature_norm)  # 各テキストとの内積（コサイン類似度）

        return similarities