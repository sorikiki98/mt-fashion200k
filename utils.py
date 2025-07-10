import os
import torch
from torch import nn


def save_model(model_path: str, cur_epoch: int, model_to_save: nn.Module):
    folder_path = os.path.dirname(model_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, model_path)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
