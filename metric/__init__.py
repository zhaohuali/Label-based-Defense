
import torch

from kernels.utils import simple_total_variation

def get_target_state(x_true):

    TV_diff = simple_total_variation(x_true)
    x_norm_loss = torch.norm(x_true.view(x_true.size(0), -1), p=2, dim=-1).mean()
    print(f'[Target] > [TV] {TV_diff:.4f} [input norm] {x_norm_loss:.4f}')