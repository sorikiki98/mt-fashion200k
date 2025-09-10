from pathlib import Path
import json


json_path = "/home/gpuadmin/dasol/mt-fashion200k/mt_dataset"

with open(Path(json_path) / "test_convergence.json",
          encoding="utf-8") as convergence:
    convergence = json.load(convergence)
#with open(Path(json_path) / "train_rollback.json", encoding="utf-8") as rollback:
#    rollback = json.load(rollback)
#with open(Path(json_path) / "train_combination.json", encoding="utf-8") as combination:
#    combination = json.load(combination)

transactions = convergence
img_ids = list()

for sample in transactions:
    n_turns = sample["n_turns"]
    tar_captions, mods = [], []

    for i in range(1, 6):  # turn-1 ~ turn-5
        if i <= n_turns:
            tar_key = f"turn-{i}"
            src_img_id = sample[tar_key]["source_img_id"]
            tar_img_id = sample[tar_key]["target_img_id"]
        if src_img_id not in img_ids:
            img_ids.append(src_img_id)
        if tar_img_id not in img_ids:
            img_ids.append(tar_img_id)

print(len(img_ids))