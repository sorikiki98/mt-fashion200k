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
from utils import AverageMeter, save_model


def train_compose(cfg, **kwargs):
    device = kwargs["device"]
    time_str = datetime.now().strftime("%m-%d-%H")
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=cfg["blip_model_name"], model_type="pretrain", is_eval=False, device=device
    )
    preprocess = targetpad_transform(cfg["target_ratio"], cfg["input_dim"])
    dataset = ComposeDataset(split="train", preprocess=preprocess, dataset_name=cfg["dataset"], cfg=cfg)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        shuffle=True
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
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(cfg["num_epochs"]):
        losses = AverageMeter()
        for idx, (
                n_turns,
                ref_img,
                tar_imgs,
                mods
        ) in enumerate(
            tqdm(
                dataloader,
                desc=f"Train [{epoch}]"
            )
        ):
            optimizer.zero_grad()
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

            if cfg["dataset"] == "200k":
                try:
                    with torch.cuda.amp.autocast():
                        loss = blip_model(
                            {"n_turns": n_turns,
                             "ref_img": ref_img,
                             "mod1": mod1_inputs,
                             "tar1_img": tar1_img,
                             "mod2": mod2_inputs,
                             "tar2_img": tar2_img,
                             "mod3": mod3_inputs,
                             "tar3_img": tar3_img,
                             "mod4": mod4_inputs,
                             "tar4_img": tar4_img,
                             "mod5": mod5_inputs,
                             "tar5_img": tar5_img,
                             }
                        )
                        print(loss)
                except Exception as e:
                    print("❌ error occurred:", e)
                    torch.cuda.empty_cache()
                    raise
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.update(loss.detach().cpu().item())
        if epoch % cfg["validation_frequency"] == 0:
            print(
                "Train Epoch: [{0}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    loss=losses,
                )
            )
            model_path = cfg["model_path"]
            save_model(f"{model_path}/{time_str}/epoch{epoch}.pth", epoch, blip_model)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["dataset"] == "200k":
        train_compose(config, device=device)
