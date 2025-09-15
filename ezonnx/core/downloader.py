from pathlib import Path
from typing import Union, Optional
from huggingface_hub import hf_hub_download
import time

_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

def get_weights(repo_id: str, filename: str, cache_dir: Optional[Path]= None) -> str:
    """
    Hugging Face Hub からウェイトを取得し、ローカルにキャッシュする。
    新しくダウンロードされた場合のみファイルパスを表示する。
    """
    cache_dir = cache_dir or _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    weight_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        # local_dir_use_symlinks=False,
        local_dir=str(cache_dir / repo_id.replace("/", "_")),
    )
    file_mtime = Path(weight_path).stat().st_mtime

    if Path(weight_path).exists() and Path(weight_path).stat().st_size != 0:
        # 新しくダウンロードされた場合のみ表示
        if file_mtime >= start_time:
            print(f"ONNX path: {weight_path}")
        return weight_path
    else:
        raise FileNotFoundError(f"Failed to download weights: {weight_path}")

if __name__ == "__main__":
    # Example usage
    repo_id = f"onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX"
    filename = "onnx/model_q4.onnx"
    weight_path = get_weights(repo_id, filename)
    print(f"Downloaded weights to: {weight_path}")