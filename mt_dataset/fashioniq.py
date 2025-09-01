from torch.utils.data import Dataset
import json
from pathlib import Path
from transformers import BertTokenizer
import PIL.Image
import torch


class MTFashionIQ(Dataset):
    def __init__(self, split, img_preprocess, txt_preprocess, category, mode, cfg):
        self.split = split
        self.img_preprocess = img_preprocess
        self.txt_preprocess = txt_preprocess
        self.img_dir = cfg["data_path"] + "/images/"
        transactions_file = cfg["data_path"] + f"/data/{category}.{split}.json"
        attributes_file = cfg["data_path"] + f"/attr/asin2attr.{category}.{split}.new.json"
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        self.max_cap_token_len = cfg["max_cap_token_len"]
        self.max_mod_token_len = cfg["max_mod_token_len"]
        self.mode = mode
        self.max_turn = 4

        if split == "train":
            with open(transactions_file, 'r', encoding='utf-8') as f:
                transactions = json.load(f)
            self.transactions = transactions

        if split == "test":
            image_captions = dict()
            with open(attributes_file, 'r', encoding='utf-8') as f:
                attributes = json.load(f)

            for key, value_list in attributes.items():
                first_elements = []
                for sublist in value_list:
                    if sublist:
                        first_elements.append(sublist[0])
                result = ' '.join(first_elements[:6])
                image_captions[key] = str(result)

            self.captions = image_captions

    def __len__(self):
        if self.mode == "relative":
            return len(self.transactions)
        elif self.mode == "classic":
            return len(self.captions)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

    def __getitem__(self, idx):
        if self.mode == "relative":
            transaction = self.transactions[idx]
            n_turns = len(transaction["reference"])

            tar_path = str(Path(self.img_dir) / transaction["target"][1])
            try:
                tar_img = self.img_preprocess(PIL.Image.open(tar_path))
            except Exception as e:
                print(f"[Index {idx}] Failed to load ref image: {tar_path} - {e}")
                raise

            ref_imgs = []
            ref_paths = []

            all_mod_input_ids = []
            all_mod_attn_mask = []
            all_ref_input_ids = []
            all_ref_attn_mask = []

            dummy_img = torch.zeros(3, self.img_preprocess.transforms[1].size, self.img_preprocess.transforms[1].size)

            tar_img_name = transaction["target"][1]
            tar_path = str(Path(self.img_dir) / tar_img_name)
            tar_cap = self.captions[tar_img_name]
            tar_cap = self.txt_preprocess(tar_cap)
            tar_cap_tokenized = self.tokenizer(
                tar_cap,
                padding="max_length",
                truncation=True,
                max_length=self.max_cap_token_len,
                return_tensors="pt",
            )
            tar_input_ids = tar_cap_tokenized["input_ids"].squeeze(0)
            tar_attn_mask = tar_cap_tokenized["attention_mask"].squeeze(0)

            for i in range(n_turns - 1, -1, -1):  # turn-1 ~ turn-5
                ref_img_name = transaction["reference"][i][2]
                ref_path = str(Path(self.img_dir) / ref_img_name)
                try:
                    ref_img = self.img_preprocess(PIL.Image.open(ref_path))
                except Exception as e:
                    print(f"[Index {idx}] Failed to load target image {ref_path}: {e}")
                    raise

                ref_imgs.append(ref_img)
                ref_paths.append(ref_path)

                ref_mod = " and ".join(transaction["reference"][i][1])
                ref_mod = self.txt_preprocess(ref_mod)
                ref_cap = self.captions[ref_img_name]
                ref_cap = self.txt_preprocess(ref_cap)

                mod_tokenized = self.tokenizer(
                    ref_mod,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_mod_token_len,
                    return_tensors="pt",
                )
                mod_inputs_ids = mod_tokenized["input_ids"].squeeze(0)
                all_mod_input_ids.append(mod_inputs_ids)
                mod_attn_mask = mod_tokenized["attention_mask"].squeeze(0)
                all_mod_attn_mask.append(mod_attn_mask)

                ref_cap_tokenized = self.tokenizer(
                    ref_cap,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_cap_token_len,
                    return_tensors="pt",
                )
                ref_input_ids = ref_cap_tokenized["input_ids"].squeeze(0)
                all_ref_input_ids.append(ref_input_ids)
                ref_attn_mask = ref_cap_tokenized["attention_mask"].squeeze(0)
                all_ref_attn_mask.append(ref_attn_mask)
            res_turns = self.max_turn - n_turns
            for _ in range(res_turns):
                ref_imgs.append(dummy_img)
                ref_paths.append("")
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

                ref_cap_tokenized = self.tokenizer(
                    "none",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_cap_token_len,
                    return_tensors="pt",
                )
                ref_input_ids = ref_cap_tokenized["input_ids"].squeeze(0)
                all_ref_input_ids.append(ref_input_ids)
                ref_attn_mask = ref_cap_tokenized["attention_mask"].squeeze(0)
                all_ref_attn_mask.append(ref_attn_mask)
            if self.split == "train":
                return {
                    "n_turns": n_turns,
                    "pil_images": ref_imgs + [tar_img],
                    "mod_input_ids": all_mod_input_ids,
                    "mod_attention_mask": all_mod_attn_mask,
                    "cap_input_ids": all_ref_input_ids + [tar_input_ids],
                }
            else:
                return {
                    "n_turns": n_turns,
                    "pil_images": ref_imgs + [tar_img],
                    "mod_input_ids": all_mod_input_ids,
                    "mod_attention_mask": all_mod_attn_mask,
                    "cap_input_ids": all_ref_input_ids + [tar_input_ids],
                    "image_paths": ref_paths + [tar_path],
                }


        elif self.mode == "classic":
            image_path = str(Path(self.img_dir) / self.captions[idx][0])
            image_name = image_path
            image = self.img_preprocess(PIL.Image.open(image_path))
            caption = self.captions[idx][1]
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
