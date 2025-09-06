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
from transformers import BertTokenizer

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
    def __init__(self, split, img_preprocess, txt_preprocess, dataset_name, mode, stage, cfg):
        self.split = split
        self.img_preprocess = img_preprocess
        self.txt_preprocess = txt_preprocess
        self.dataset_name = dataset_name.lower()
        self.json_file_root = cfg["json_path"]
        self.data_root = cfg["data_path"]
        self.max_cap_token_len = cfg["max_cap_token_len"]
        self.max_mod_token_len = cfg["max_mod_token_len"]
        self.model_name = cfg["blip_model_name"]
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")

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
            if stage == "convergence":
                self.transactions = convergence
            elif stage == "rollback":
                self.transactions = rollback
            elif stage == "combination":
                self.transactions = combination
            else:
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
            probs = transaction["probs"]
            ref_path = str(Path(self.data_root) / transaction["turn-1"]["source_img_path"])
            try:
                ref_img = self.img_preprocess(PIL.Image.open(ref_path))
            except Exception as e:
                print(f"[Index {index}] Failed to load ref image: {ref_path} - {e}")
                raise

            tar_imgs = []
            tar_paths = []

            all_mod_input_ids = []
            all_mod_attn_mask = []
            all_tar_input_ids = []
            all_tar_attn_mask = []

            dummy_img = torch.zeros(3, self.img_preprocess.transforms[1].size, self.img_preprocess.transforms[1].size)
            dummy_input_ids = torch.zeros(self.max_cap_token_len, dtype=torch.long)
            dummy_attention_mask = torch.zeros(self.max_cap_token_len, dtype=torch.long)

            ref_caption = transaction["turn-1"]["source_caption"]
            ref_cap_tokenized = self.tokenizer(
                ref_caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_cap_token_len,
                return_tensors="pt",
            )
            ref_input_ids = ref_cap_tokenized["input_ids"].squeeze(0)
            ref_attn_mask = ref_cap_tokenized["attention_mask"].squeeze(0)

            rollback_input_ids = dummy_input_ids
            rollback_attn_mask = dummy_attention_mask
            rollback_img = dummy_img
            combination_input_ids = dummy_input_ids
            combination_attn_mask = dummy_attention_mask

            is_rollback = transaction["is_rollback"]
            is_combination = transaction["is_combination"]

            for i in range(1, 6):  # turn-1 ~ turn-5
                if i <= n_turns:
                    tar_key = f"turn-{i}"
                    tar_path = str(Path(self.data_root) / transaction[tar_key]["target_img_path"])
                    try:
                        tar_img = self.img_preprocess(PIL.Image.open(tar_path))
                    except Exception as e:
                        print(f"[Index {index}] Failed to load target image {tar_path}: {e}")
                        raise
                    mod = transaction[tar_key]["mod_str"]
                    mod = self.txt_preprocess(mod)
                    tar_caption = transaction[tar_key]["target_caption"]
                    tar_caption = self.txt_preprocess(tar_caption)

                    tar_imgs.append(tar_img)
                    tar_paths.append(tar_path)

                    mod_tokenized = self.tokenizer(
                        mod,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_mod_token_len,
                        return_tensors="pt",
                    )
                    mod_inputs_ids = mod_tokenized["input_ids"].squeeze(0)
                    all_mod_input_ids.append(mod_inputs_ids)
                    mod_attn_mask = mod_tokenized["attention_mask"].squeeze(0)
                    all_mod_attn_mask.append(mod_attn_mask)

                    tar_cap_tokenized = self.tokenizer(
                        tar_caption,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_cap_token_len,
                        return_tensors="pt",
                    )
                    tar_input_ids = tar_cap_tokenized["input_ids"].squeeze(0)
                    all_tar_input_ids.append(tar_input_ids)
                    tar_attn_mask = tar_cap_tokenized["attention_mask"].squeeze(0)
                    all_tar_attn_mask.append(tar_attn_mask)
                    if i == n_turns:
                        tar_key = f"turn-{i}"
                        if is_rollback:
                            if "rollback_img_path" in transaction[tar_key] and "rollback_caption" in transaction[
                                tar_key]:
                                rollback_img_path = str(
                                    Path(self.data_root) / transaction[tar_key]["rollback_img_path"])
                                try:
                                    rollback_img = self.img_preprocess(PIL.Image.open(rollback_img_path))
                                except Exception as e:
                                    print(f"[Index {index}] Failed to load target image {rollback_img_path}: {e}")
                                    raise
                                rollback_caption = transaction[tar_key]["rollback_caption"]
                                rollback_cap_tokenized = self.tokenizer(
                                    rollback_caption,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.max_cap_token_len,
                                    return_tensors="pt",
                                )
                                rollback_input_ids = rollback_cap_tokenized["input_ids"].squeeze(0)
                                rollback_attn_mask = rollback_cap_tokenized["attention_mask"].squeeze(0)
                        elif is_combination:
                            if "combination_caption" in transaction[tar_key]:
                                combination_caption = transaction[tar_key]["combination_caption"]
                                combination_cap_tokenized = self.tokenizer(
                                    combination_caption,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.max_cap_token_len,
                                    return_tensors="pt",
                                )
                                combination_input_ids = combination_cap_tokenized["input_ids"].squeeze(0)
                                combination_attn_mask = combination_cap_tokenized["attention_mask"].squeeze(0)
                else:
                    tar_imgs.append(dummy_img)
                    tar_paths.append("")
                    mod_tokenized = self.tokenizer(
                        "none",
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_mod_token_len,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    mod_inputs_ids = mod_tokenized["input_ids"].squeeze(0)
                    all_mod_input_ids.append(mod_inputs_ids)
                    mod_attn_mask = mod_tokenized["attention_mask"].squeeze(0)
                    all_mod_attn_mask.append(mod_attn_mask)

                    tar_cap_tokenized = self.tokenizer(
                        "none",
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_cap_token_len,
                        return_tensors="pt",
                    )
                    tar_input_ids = tar_cap_tokenized["input_ids"].squeeze(0)
                    all_tar_input_ids.append(tar_input_ids)
                    tar_attn_mask = tar_cap_tokenized["attention_mask"].squeeze(0)
                    all_tar_attn_mask.append(tar_attn_mask)

            if self.split == "train":
                return {
                    "n_turns": n_turns,
                    "pil_images": [ref_img] + tar_imgs,
                    "mod_input_ids": all_mod_input_ids,
                    "mod_attention_mask": all_mod_attn_mask,
                    "cap_input_ids": [ref_input_ids] + all_tar_input_ids,
                    "cap_attention_mask": [ref_attn_mask] + all_tar_attn_mask,
                    "probs": probs,
                    "is_rollback": is_rollback,
                    "is_combination": is_combination,
                    "rollback_input_ids": rollback_input_ids,  # for MAI
                    "rollback_attention_mask": rollback_attn_mask,  # for MAI
                    "rollback_images": rollback_img,  # for MAI
                    "combination_input_ids": combination_input_ids,  # for MAI
                    "combination_attention_mask": combination_attn_mask  # for MAI
                }
            else:
                return {
                    "n_turns": n_turns,
                    "pil_images": [ref_img] + tar_imgs,
                    "mod_input_ids": all_mod_input_ids,
                    "mod_attention_mask": all_mod_attn_mask,
                    "cap_input_ids": [ref_input_ids] + all_tar_input_ids,
                    "cap_attention_mask": [ref_attn_mask] + all_tar_attn_mask,
                    "probs": probs,
                    "image_paths": [ref_path] + tar_paths,
                    "is_rollback": is_rollback,
                    "is_combination": is_combination,
                    "rollback_input_ids": rollback_input_ids,  # for MAI
                    "rollback_attention_mask": rollback_attn_mask,  # for MAI
                    "rollback_images": rollback_img,  # for MAI
                    "combination_input_ids": combination_input_ids,  # for MAI
                    "combination_attention_mask": combination_attn_mask  # for MAI
                }
        elif self.mode == "classic":
            image_path = str(Path(self.data_root) / self.image_names[index])
            image_name = image_path
            image = self.img_preprocess(PIL.Image.open(image_path))
            caption = self.image_captions[index]
            cap_tokenized = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_cap_token_len,
                return_tensors="pt",
            )
            cap_input_ids = cap_tokenized["input_ids"].squeeze(0)
            cap_attn_mask = cap_tokenized["attention_mask"].squeeze(0)
            return image_name, image, cap_input_ids, cap_attn_mask
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
