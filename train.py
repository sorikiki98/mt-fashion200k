import json
import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from mt_dataset import targetpad_transform, ComposeDataset
from utils import AverageMeter, save_model, setup_seed
from retrospection import RetrospectiveMultiTurnCirModel


def train(cfg, **kwargs):
    device = kwargs["device"]
    stage = kwargs["stage"]
    model_name = cfg["blip_model_name"]

    if stage != "convergence" and stage != "combination" and stage != "rollback" and stage != "retrospective":
        raise ValueError("Stage should be in ['convergence', 'combination', 'rollback', 'retrospective']")

    blip_model, _, txt_processors = load_model_and_preprocess(
        name=model_name, model_type="pretrain", is_eval=False, device=device
    )

    if model_name == "blip2_qformer_cir_align_convergence" and stage == "retrospective":
        model = RetrospectiveMultiTurnCirModel(blip_model, cfg["max_mod_token_len"], cfg["max_turn"])
        model.to(device)
    else:
        model = blip_model

    start_epoch = 0
    if "resume_path" in cfg and cfg["resume_path"]:
        print(f"Loading checkpoint from epoch{cfg['resume_path']}")
        checkpoint = torch.load(cfg["resume_path"], map_location=device)  # todo

        model_key = "Blip2QformerGatedAttention"
        if model_key in checkpoint:
            state_dict = checkpoint[model_key]

            state_dict['Qformer.cls.predictions.bias'] = state_dict['Qformer.cls.predictions.bias'][:30522]
            state_dict['bertLM.cls.predictions.bias'] = state_dict['bertLM.cls.predictions.bias'][:30522]

            missing_keys, unexpected_keys = blip_model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded state_dict from key '{model_key}'")
            if missing_keys:
                print("Missing keys:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys:", unexpected_keys)
        else:
            print(f"Key '{model_key}' not found in checkpoint. Available keys: {list(checkpoint.keys())}")

        """
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Warning: No epoch info found in checkpoint. Starting from epoch 0.")
        """

    img_preprocessors = targetpad_transform(cfg["target_ratio"], cfg["input_dim"])
    dataset = ComposeDataset(split="train", img_preprocess=img_preprocessors, txt_preprocess=txt_processors["eval"],
                             dataset_name=cfg["dataset"], mode="relative",
                             stage=stage, cfg=cfg)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        persistent_workers=True
    )

    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, model.parameters()),
                "lr": cfg["learning_rate"],
                "betas": (0.9, 0.98),
                "eps": 1e-7,
                "weight_decay": 0.05,
            }
        ]
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["learning_rate"],
        pct_start=6 / cfg["num_epochs"],
        div_factor=100.0,
        steps_per_epoch=len(dataloader),
        epochs=cfg["num_epochs"],
    )
    if start_epoch > 0:
        scheduler.last_epoch = start_epoch * len(dataloader) - 1

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(start_epoch, cfg["num_epochs"]):
        losses = AverageMeter()
        for idx, samples in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Train [{epoch}]"
                )
        ):
            optimizer.zero_grad(set_to_none=True)
            model.train()

            images = samples.get("pil_images")
            mod_input_ids = samples.get("mod_input_ids")
            mod_attention_mask = samples.get("mod_attention_mask")
            cap_input_ids = samples.get("cap_input_ids")
            cap_attention_mask = samples.get("cap_attention_mask")
            n_turns = samples.get("n_turns")

            is_rollback = samples.get("is_rollback")
            is_combination = samples.get("is_combination")

            rollback_input_ids = samples.get("rollback_input_ids")
            rollback_attention_mask = samples.get("rollback_attention_mask")
            rollback_images = samples.get("rollback_images")
            combination_input_ids = samples.get("combination_input_ids")
            combination_attention_mask = samples.get("combination_attention_mask")

            if cfg["dataset"] == "200k":
                try:
                    with torch.cuda.amp.autocast():
                        loss = model(
                            {"n_turns": n_turns,
                             "images": images,
                             "mod_input_ids": mod_input_ids,
                             "mod_attention_mask": mod_attention_mask,
                             "cap_input_ids": cap_input_ids,
                             "cap_attention_mask": cap_attention_mask,
                             "is_rollback": is_rollback,
                             "is_combination": is_combination,
                             "rollback_input_ids": rollback_input_ids,
                             "rollback_attention_mask": rollback_attention_mask,
                             "rollback_images": rollback_images,
                             "combination_input_ids": combination_input_ids,
                             "combination_attention_mask": combination_attention_mask
                             }
                        )

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        losses.update(loss.detach().cpu().item())
                except Exception as e:
                    print("❌ error occurred:", e)
                    raise
        if epoch % cfg["validation_frequency"] == 0:
            print(
                "Train Epoch: [{0}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    loss=losses,
                )
            )
            model_path = cfg["model_path"]
            save_model(f"{model_path}/epoch{epoch}.pth", epoch, model)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    setup_seed(42)

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["dataset"] == "200k":
        train(config, stage="retrospective", device=device)
