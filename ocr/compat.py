"""
PyTorch 2.6+ 호환성 패치

torch.load()의 weights_only 기본값이 True로 변경되면서 기존 모델 로드가
실패하는 문제를 해결합니다. 이 모듈을 임포트하면 패치가 자동 적용됩니다.

사용법:
    import ocr.compat  # noqa: F401
"""
import torch

_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, "weights_only": False})
