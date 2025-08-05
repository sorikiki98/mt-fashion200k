import os
import torch
from torch import nn
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def save_model(model_path: str, cur_epoch: int, model_to_save: nn.Module):
    folder_path = os.path.dirname(model_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, model_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def extract_index_blip_fusion_features(dataset, model):
    classic_val_loader = DataLoader(
        dataset=dataset,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    index_fusion_features = []
    index_names = []

    for names, images, caption_input_ids, caption_attention_mask in tqdm(classic_val_loader, desc="Index"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            index_fusion_feats = model.extract_target_features(images,
                                                               caption_input_ids,
                                                               caption_attention_mask)
            index_fusion_features.append(index_fusion_feats)
            index_names.extend(names)

    index_fusion_features = torch.vstack(index_fusion_features)
    index_names = [str(os.path.dirname(name)) for name in index_names]
    return index_fusion_features, index_names


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


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


def validate_list_of_strings(name, lst):
    if not isinstance(lst, list):
        print(f"❌ {name} is not a list: {type(lst)}")
        return False
    for j, item in enumerate(lst):
        if not isinstance(item, str):
            print(f"❌ {name}[{j}] is not str: {item} ({type(item)})")
            return False
        if item.strip() == "":
            print(f"⚠️ {name}[{j}] is an empty string.")
    return True
