import pytest
from ezonnx import *
import numpy as np

@pytest.mark.parametrize("model_class, model_args", [
    (RTMDet, {"identifier": "m-person"}),
    (RTMPose, {"identifier": "l"}),
    (RTMW, {"identifier": "l-384"}),
    (DWPose, {"identifier": "ll"}),
    (RTMO, {"identifier": "s"}),
    (RFDETR, {"identifier": "s"}),
    (DFINE, {"identifier": "s"}),
    (ViTPose, {"identifier": "small"}),
])
def test_detection(model_class, model_args):
    model = model_class(**model_args)
    result = model("examples/images/surf.jpg")
    assert hasattr(result, 'boxes')
    assert hasattr(result, 'scores')
    assert result.boxes.shape[1]==4
    assert len(result.boxes)==len(result.scores)
    assert len(result.scores)>=1 and len(result.scores)<=10
    assert result.scores[0]<=1 and result.scores[0]>0
    
    if hasattr(result, 'classes'):
        assert hasattr(result, 'classes')
        assert len(result.classes)==len(result.scores)
        assert len(result.classes)>=1
        assert isinstance(result.classes[0], (np.integer, int))
    if hasattr(result, 'kpts'):
        assert len(result.kpts.shape)==3
        assert len(result.kpts)==1
    assert hasattr(result, 'visualized_img')
    assert result.visualized_img.shape==result.original_img.shape
    del model
    del result
