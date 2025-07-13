import os
import json
import sys

from mt_dataset import ComposeDataset, targetpad_transform

import torch

from lavis.models import load_model_and_preprocess
from utils import setup_seed, extract_index_blip_fusion_features
from validate import compute_blip_compose_multi


def test_compose(cfg, **kwargs):
    device = kwargs["device"]
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=cfg["blip_model_name"], model_type="pretrain", is_eval=False, device=device
    )
    try:
        checkpoint = torch.load(cfg["resume_path"], map_location=device)
    except Exception as e:
        print("❌ Failed to load:", e)
    blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    preprocess = targetpad_transform(cfg["target_ratio"], cfg["input_dim"])

    relative_val_dataset = ComposeDataset(split="test", preprocess=preprocess, dataset_name=cfg["dataset"],
                                          mode="relative", cfg=cfg)
    classic_val_dataset = ComposeDataset(split="test", preprocess=preprocess, dataset_name=cfg["dataset"],
                                         mode="classic", cfg=cfg)

    with torch.no_grad():
        blip_model.eval()
        index_feats, index_names = extract_index_blip_fusion_features(classic_val_dataset, blip_model)

        recall_dict = compute_blip_compose_multi(
            relative_val_dataset, blip_model, index_feats, index_names, txt_processors, cfg["dataset"]
        )

        for k, (recall_at1, recall_at5, recall_at10, recall_at20) in recall_dict.items():
            print(k, "*" * 40)
            r_average = (recall_at1 + recall_at5 + recall_at10 + recall_at20) / 4
            print("R@1:", recall_at1, "  R@5: ", recall_at5, "  R@10:", recall_at10, "  R@20: ", recall_at20)
            print("Mean Now: ", r_average, "*" * 30)


if __name__ == '__main__':
    setup_seed(42)

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["dataset"] == "200k":
        test_compose(config, device=device)
