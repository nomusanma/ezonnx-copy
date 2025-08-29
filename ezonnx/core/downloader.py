from pathlib import Path
from typing import Union, Optional
from huggingface_hub import hf_hub_download

_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

def get_weights(repo_id: str, filename: str, cache_dir: Optional[Path]= None) -> str:
    """
    Hugging Face Hub からウェイトを取得し、ローカルにキャッシュする。
    """
    cache_dir = cache_dir or _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir)
    )
    print(f"ONNX path: {weight_path}")
    return weight_path

if __name__ == "__main__":
    # Example usage
    repo_id = f"onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX"
    filename = "onnx/model_q4.onnx"
    weight_path = get_weights(repo_id, filename)
    print(f"Downloaded weights to: {weight_path}")