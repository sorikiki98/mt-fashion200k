import os
import json
import sys

from mt_dataset import ComposeDataset, targetpad_transform

import torch

from lavis.models import load_model_and_preprocess
from utils import setup_seed, extract_index_blip_fusion_features
from validate import visualize_result_for_single_transaction, save_top_k_retrieval_results
from retrospection import RetrospectiveMultiTurnCirModel

from PIL import Image
from pathlib import Path

def deploy(cfg, **kwargs):
    device = kwargs["device"]
    stage = kwargs["stage"]
    sample_idx = kwargs["sample_idx"]

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

    with (torch.no_grad()):
        blip_model.eval()
        index_feats, index_names = extract_index_blip_fusion_features(classic_val_dataset, model)

        for idx in sample_idx:
            first_sim, second_sim, last_sim, first_target, second_target, last_target = \
            visualize_result_for_single_transaction(model, relative_val_dataset, index_feats, idx)
            save_top_k_retrieval_results(
                first_similarity=first_sim,
                second_similarity=second_sim,
                last_similarity=last_sim,
                first_target_name=first_target,
                second_target_name=second_target,
                last_target_name=last_target,
                index_names=index_names,
                sample_idx=idx,
                output_dir="visualization_sample",
                top_k=20,
            )

if __name__ == '__main__':
    setup_seed(42)

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["dataset"] == "200k":
        deploy(config, stage="convergence", device=device, sample_idx=[2761, 2762, 2763, 2764])
