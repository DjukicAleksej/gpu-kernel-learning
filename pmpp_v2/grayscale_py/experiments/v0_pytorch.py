#!POPCORN leaderboard grayscale_v2
#!POPCORN gpu A100

"""v0: official PMPP_v2 PyTorch starter used only as the measured baseline."""

import torch

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    image, output = data
    weights = torch.tensor(
        [0.2989, 0.5870, 0.1140],
        device=image.device,
        dtype=image.dtype,
    )
    output[...] = torch.sum(image * weights, dim=-1)
    return output

