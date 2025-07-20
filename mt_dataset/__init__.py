from torch.utils.data import Dataset

import json
import os
import PIL
from pathlib import Path
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch

from mt_dataset.fashion200k import Fashion200k
from mt_dataset.fashion200k_utils import extract_image_names_and_captions


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TargetPad:
    def __init__(self, target_ratio: float, size: int):
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, "constant")


def targetpad_transform(target_ratio: float, dim: int):
    return Compose(
        [
            TargetPad(target_ratio, dim),
            Resize(dim),
            CenterCrop(dim),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class ComposeDataset(Dataset):
    def __init__(self, split, preprocess, dataset_name, mode, cfg):
        self.split = split
        self.preprocess = preprocess
        self.dataset_name = dataset_name.lower()
        self.json_file_root = cfg["json_path"]
        self.data_root = cfg["data_path"]
        self.mode = mode

        if self.dataset_name == "200k":
            if split == "train":
                with open(Path(self.json_file_root) / "train_convergence.json",
                          encoding="utf-8") as convergence:
                    convergence = json.load(convergence)
                with open(Path(self.json_file_root) / "train_rollback.json", encoding="utf-8") as rollback:
                    rollback = json.load(rollback)
                with open(Path(self.json_file_root) / "train_combination.json", encoding="utf-8") as combination:
                    combination = json.load(combination)
            else:
                with open(Path(self.json_file_root) / "test_convergence.json",
                          encoding="utf-8") as convergence:
                    convergence = json.load(convergence)
                with open(Path(self.json_file_root) / "test_rollback.json", encoding="utf-8") as rollback:
                    rollback = json.load(rollback)
                with open(Path(self.json_file_root) / "test_combination.json", encoding="utf-8") as combination:
                    combination = json.load(combination)
            self.transactions = convergence + rollback + combination
            if split == "test":
                self.image_names, self.image_captions = extract_image_names_and_captions(self.data_root, split)

    def __len__(self):
        if self.mode == "relative":
            return len(self.transactions)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    def __getitem__(self, index):
        if self.mode == "relative":
            transaction = self.transactions[index]
            n_turns = transaction["n_turns"]

            ref_path = Path(self.data_root) / transaction["turn-1"]["source_img_path"]
            try:
                ref_img = self.preprocess(PIL.Image.open(ref_path))
            except Exception as e:
                print(f"[Index {index}] Failed to load ref image: {ref_path} - {e}")
                raise

            tar_imgs, tar_captions, mods = [], [], []
            tar_paths = []
            dummy_img = torch.zeros(3, self.preprocess.transforms[1].size, self.preprocess.transforms[1].size)
            ref_caption = transaction["turn-1"]["source_caption"]

            for i in range(1, 6):  # turn-1 ~ turn-5
                if i <= n_turns:
                    tar_key = f"turn-{i}"
                    tar_path = str(Path(self.data_root) / transaction[tar_key]["target_img_path"])
                    try:
                        tar_img = self.preprocess(PIL.Image.open(tar_path))
                    except Exception as e:
                        print(f"[Index {index}] Failed to load target image {tar_path}: {e}")
                        raise
                    mod = transaction[tar_key]["mod_str"]
                    tar_caption = transaction[tar_key]["target_caption"]

                    tar_imgs.append(tar_img)
                    tar_paths.append(tar_path)
                    tar_captions.append(tar_caption)
                    mods.append(mod)
                else:
                    tar_imgs.append(dummy_img)
                    tar_paths.append("")
                    tar_captions.append("none")
                    mods.append("none")
            if self.split == "train":
                return (
                    n_turns,
                    ref_img,  # (C, H, W)
                    ref_caption,
                    torch.stack(tar_imgs),  # (5, C, H, W),
                    tar_captions,  # [str x 5]
                    mods  # [str x 5]
                )
            else:
                return (
                    n_turns,
                    ref_img,
                    str(ref_path),
                    ref_caption,
                    torch.stack(tar_imgs),
                    tar_paths,
                    tar_captions,
                    mods
                )
        elif self.mode == "classic":
            image_path = str(Path(self.data_root) / self.image_names[index])
            image_name = image_path
            image = self.preprocess(PIL.Image.open(image_path))
            caption = self.image_captions[index]
            return image_name, image, caption
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
