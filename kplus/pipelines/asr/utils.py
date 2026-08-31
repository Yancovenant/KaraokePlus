import torch

__all__ = [
    "get_default_dtype",
]

def get_default_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32

