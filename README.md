# EZONNX
Python library for easily handling state-of-the-art ONNX models.  
Simple and unified API.
<div align="center"><img src=./assets/top.jpg  width=800/> </div>

```python
from ezonnx import DinoV3
model = DinoV3('vits16',size=768) # download & create model
res = model("image.jpg")          # inference
print(res.visualized_img)         # get visualized image
# access outputs
print(res.class_token, res.patch_tokens, res.pca_img_rgb)
```

## 📦 Model Collection

- Image classification  
    - Siglip2 (zero-shot classification)
- Object detection
    - D-FINE
    - RTMDet (Person detection)
- Segmentation
    - SAM2
    - YOLO-seg
- Feature extraction
    - DINOv2, DINOv3
    - Siglip2
- Person pose estimation
    - DWPose (whole body)
- Depth estimation
    - DepthAnythingV2
- Background removal
    - RMBG1.4
- Image inpainting
    - LaMa
- Optical flow
    - NeuFlowV2
---

## 🛠️ Setup
Recommend to use uv.
Requires Python 3.12 or later. Dependencies are managed via `pyproject.toml`

```sh
git clone https://github.com/yourname/ezonnx.git
cd ezonnx
uv init
uv sync
```

---

## ⚡ Quick Start

Ready-to-use sample code for image classification and object detection.

```python
from ezonnx import DinoV2,DinoV3,visualize_images

img_path = "images/cat.jpg"

dinov3 = DinoV3("vits16", size=768)
dinov2 = DinoV2("small", size=768)

out_v3 = dinov3(img_path)
out_v2 = dinov2(img_path)

visualize_images(["Original image","DinoV3 PCA","DinoV2 PCA"], 
                [out_v3.original_img[...,::-1],out_v3.visualized_img, out_v2.visualized_img])
```
<img src=./assets/quickstart.jpg  width=800/> 

Find more notebook samples in the [examples/](examples/) folder.

---

## 📚 Examples

- [examples/feature_extraction.ipynb](examples/feature_extraction.ipynb)
- [examples/object_detection.ipynb](examples/object_detection.ipynb)
- [examples/image_processing.ipynb](examples/image_processing.ipynb)
- [examples/segmentation.ipynb](examples/segmentation.ipynb)

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.

---


## 🙏 Acknowledgements  

Special thanks to Hugging Face and the developers of each AI model integrated in EZONNX. Their open-source contributions and innovative research make it possible to provide powerful, easy-to-use tools for the community.