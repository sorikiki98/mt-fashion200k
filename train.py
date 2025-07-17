import json
import sys
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from datetime import datetime
from lavis.models import load_model_and_preprocess
from mt_dataset import targetpad_transform, ComposeDataset
from utils import AverageMeter, save_model, setup_seed


def train_compose(cfg, **kwargs):
    device = kwargs["device"]
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=cfg["blip_model_name"], model_type="pretrain", is_eval=False, device=device
    )

    start_epoch = 0
    if "resume_path" in cfg and cfg["resume_path"]:
        print(f"Loading checkpoint from epoch{cfg['resume_path']}")
        checkpoint = torch.load(cfg["resume_path"], map_location=device)

        model_key = blip_model.__class__.__name__
        if model_key in checkpoint:
            missing_keys, unexpected_keys = blip_model.load_state_dict(checkpoint[model_key], strict=False)
            print(f"Successfully loaded state_dict from key '{model_key}'")
            if missing_keys:
                print("Missing keys:", missing_keys)
            if unexpected_keys:
                print("Unexpected keys:", unexpected_keys)
        else:
            print(f"Key '{model_key}' not found in checkpoint. Available keys: {list(checkpoint.keys())}")

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Warning: No epoch info found in checkpoint. Starting from epoch 0.")

    preprocess = targetpad_transform(cfg["target_ratio"], cfg["input_dim"])
    dataset = ComposeDataset(split="train", preprocess=preprocess, dataset_name=cfg["dataset"], mode="relative",
                             cfg=cfg)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        persistent_workers=True,
    )
    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, blip_model.parameters()),
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
        for idx, (
                n_turns,
                ref_img,
                ref_cap,
                tar_imgs,
                tar_caps,
                mods
        ) in enumerate(
            tqdm(
                dataloader,
                desc=f"Train [{epoch}]"
            )
        ):
            optimizer.zero_grad(set_to_none=True)
            blip_model.train()

            ref_img = ref_img.to(device, non_blocking=True)
            tar1_img = tar_imgs[:, 0].to(device, non_blocking=True)  # torch.size([32, 3, 225, 225])
            tar2_img = tar_imgs[:, 1].to(device, non_blocking=True)
            tar3_img = tar_imgs[:, 2].to(device, non_blocking=True)
            tar4_img = tar_imgs[:, 3].to(device, non_blocking=True)
            tar5_img = tar_imgs[:, 4].to(device, non_blocking=True)

            mods = list(zip(*mods))
            mod1_inputs = [txt_processors["eval"](m[0]) for m in mods]
            mod2_inputs = [txt_processors["eval"](m[1]) for m in mods]
            mod3_inputs = [txt_processors["eval"](m[2]) for m in mods]
            mod4_inputs = [txt_processors["eval"](m[3]) for m in mods]
            mod5_inputs = [txt_processors["eval"](m[4]) for m in mods]

            tar_caps = list(zip(*tar_caps))
            ref_captions = [txt_processors["eval"](cap) for cap in ref_cap]
            tar1_captions = [txt_processors["eval"](cap[0]) for cap in tar_caps]
            tar2_captions = [txt_processors["eval"](cap[1]) for cap in tar_caps]
            tar3_captions = [txt_processors["eval"](cap[2]) for cap in tar_caps]
            tar4_captions = [txt_processors["eval"](cap[3]) for cap in tar_caps]
            tar5_captions = [txt_processors["eval"](cap[4]) for cap in tar_caps]

            if cfg["dataset"] == "200k":
                try:
                    with torch.amp.autocast("cuda"):
                        loss = blip_model(
                            {"n_turns": n_turns,
                             "ref_img": ref_img,
                             "ref_cap": ref_captions,
                             "mod1": mod1_inputs,
                             "tar1_img": tar1_img,
                             "tar1_cap": tar1_captions,
                             "mod2": mod2_inputs,
                             "tar2_img": tar2_img,
                             "tar2_cap": tar2_captions,
                             "mod3": mod3_inputs,
                             "tar3_img": tar3_img,
                             "tar3_cap": tar3_captions,
                             "mod4": mod4_inputs,
                             "tar4_img": tar4_img,
                             "tar4_cap": tar4_captions,
                             "mod5": mod5_inputs,
                             "tar5_img": tar5_img,
                             "tar5_cap": tar5_captions
                             }
                        )

                        # if idx % 100 == 0:
                        #    print(f"Batch-{idx} Loss: {loss}")
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        losses.update(loss.detach().cpu().item())
                except Exception as e:
                    print("❌ error occurred:", e)
                    torch.cuda.empty_cache()
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
            save_model(f"{model_path}/epoch{epoch}.pth", epoch, blip_model)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    setup_seed(42)

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["dataset"] == "200k":
        train_compose(config, device=device)
