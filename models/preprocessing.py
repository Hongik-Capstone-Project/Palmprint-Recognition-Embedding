import torch


class NormSingleROI:
    """
    비배경(비검은) 픽셀만 평균/표준편차로 정규화.
    배경(검은 영역)은 그대로 유지.

    Input  : Tensor [1, H, W]
    Output : Tensor [1, H, W]  (outchannels=1)
           : Tensor [C, H, W]  (outchannels=C, repeat)
    """

    def __init__(self, outchannels: int = 1):
        self.outchannels = outchannels

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        c, h, w = tensor.size()

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t   = tensor[idx]

        if t.numel() > 0:
            m = t.mean()
            s = t.std()
            tensor[idx] = t.sub_(m).div_(s + 1e-6)

        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor
