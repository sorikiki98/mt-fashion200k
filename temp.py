from transformers import BertTokenizer
from pathlib import Path
import json

from lavis.models import load_model_and_preprocess

json_path = "/mnt/c/Users/user/PycharmProjects/mtcir/mt_dataset"

with open(Path(json_path) / "train_convergence.json",
          encoding="utf-8") as convergence:
    convergence = json.load(convergence)
with open(Path(json_path) / "train_rollback.json", encoding="utf-8") as rollback:
    rollback = json.load(rollback)
with open(Path(json_path) / "train_combination.json", encoding="utf-8") as combination:
    combination = json.load(combination)

transactions = convergence
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
blip_model, _, txt_processors = load_model_and_preprocess(
    name=config["blip_model_name"], model_type="pretrain", is_eval=False, device="cuda"
)

for sample in transactions:
    n_turns = sample["n_turns"]
    tar_captions, mods = [], []

    for i in range(1, 6):  # turn-1 ~ turn-5
        if i <= n_turns:
            tar_key = f"turn-{i}"
            mod = sample[tar_key]["mod_str"]
            processed_mod = txt_processors["eval"](mod)
            tar_caption = sample[tar_key]["target_caption"]
            processed_tar_caption = txt_processors["eval"](tar_caption)
            tar_captions.append(processed_tar_caption)
            mods.append(processed_mod)
        else:
            processed_none = txt_processors["eval"]("none")
            tar_captions.append(processed_none)
            mods.append(processed_none)

        turn_key = f"turn-{i}"
    try:
        cap_tokens = tokenizer(
            tar_captions,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt",
            add_special_tokens=False
        )
    except Exception as e:
        print("[Tokenizer Error] cap_tokens failed")
        print("Input tar_captions:", tar_captions)
        raise e
    try:
        mod_tokens = tokenizer(
            mods,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt",
            add_special_tokens=False
        )
    except Exception as e:
        print("[Tokenizer Error] mod_tokens failed")
        print("Input mods:", mods)
        raise e
