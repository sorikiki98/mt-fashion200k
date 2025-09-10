import os
import json
import sys

from mt_dataset import ComposeDataset, targetpad_transform

import torch

from lavis.models import load_model_and_preprocess
from utils import setup_seed, extract_index_blip_fusion_features
from validate import compute_blip_compose_multi
from retrospection import RetrospectiveMultiTurnCirModel


def test(cfg, **kwargs):
    device = kwargs["device"]
    stage = kwargs["stage"]

    if stage != "convergence" and stage != "combination" and stage != "rollback" and stage != "retrospective":
        raise ValueError("Stage should be in ['convergence', 'combination', 'rollback', 'retrospective']")

    blip_model, _, txt_processors = load_model_and_preprocess(
        name=cfg["blip_model_name"], model_type="pretrain", is_eval=False, device=device
    )

    if stage == "retrospective":
        model = RetrospectiveMultiTurnCirModel(blip_model, cfg["max_mod_token_len"], cfg["max_turn"])
        model.to(device)
    else:
        model = blip_model

    if "resume_path" in cfg and cfg["resume_path"]:
        print(f"Loading checkpoint from epoch{cfg['resume_path']}")
        checkpoint = torch.load(cfg["resume_path"], map_location=device)  # todo

        if "model" in checkpoint:
            state_dict = checkpoint["model"]

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"✓ Model state loaded from 'model' key")
            if missing_keys:
                print("Missing keys:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys:", unexpected_keys)
        elif "model_class" in checkpoint:
            model_key = checkpoint["model_class"]
            if model_key in checkpoint:
                state_dict = checkpoint[model_key]
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"✓ Model state loaded from '{model_key}' key (legacy format)")
                if missing_keys:
                    print(f"  - Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"  - Unexpected keys: {unexpected_keys}")
            else:
                print(f"⚠️ Model key '{model_key}' not found in checkpoint")
        else:
            print(f"Model not found in checkpoint. Available keys: {list(checkpoint.keys())}")

    img_preprocessors = targetpad_transform(cfg["target_ratio"], cfg["input_dim"])

    relative_val_dataset = ComposeDataset(split="test",
                                          img_preprocess=img_preprocessors,
                                          txt_preprocess=txt_processors["eval"],
                                          dataset_name=cfg["dataset"],
                                          mode="relative",
                                          stage=stage,
                                          cfg=cfg)
    classic_val_dataset = ComposeDataset(split="test",
                                         img_preprocess=img_preprocessors,
                                         txt_preprocess=txt_processors["eval"],
                                         dataset_name=cfg["dataset"],
                                         mode="classic",
                                         stage=stage,
                                         cfg=cfg)

    with torch.no_grad():
        blip_model.eval()
        index_feats, index_names = extract_index_blip_fusion_features(classic_val_dataset, model)

        recall_dict = compute_blip_compose_multi(relative_val_dataset, model, index_feats, index_names)

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
        test(config, stage="rollback", device=device)
