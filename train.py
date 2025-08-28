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
import faulthandler
faulthandler.enable()


def train(cfg, **kwargs):
    device = kwargs["device"]
    stage = kwargs["stage"]

    if stage != "convergence" and stage != "combination" and stage != "rollback" and stage != "retrospective":
        raise ValueError("Stage should be in ['convergence', 'combination', 'rollback', 'retrospective']")

    model, _, txt_processors = load_model_and_preprocess(
        name=cfg["blip_model_name"], model_type="pretrain", is_eval=False, device=device
    )

    start_epoch = 0
    global_step = 0

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

    if cfg.get("resume_path"):
        ckpt_path = cfg["resume_path"]
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "model" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            print("✓ model state loaded (missing:", missing, ", unexpected:", unexpected, ")")
        else:
            model_key = ckpt.get("model_class") or "Blip2QformerCirAlignRetrospective"
            if model_key in ckpt:
                missing, unexpected = model.load_state_dict(ckpt[model_key], strict=False)
                print(f"✓ model state loaded from '{model_key}' (missing:{missing}, unexpected:{unexpected})")
            else:
                print("⚠️ no model state found in checkpoint keys:", list(ckpt.keys()))

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("✓ optimizer state loaded")
        if "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
                print("✓ scaler state loaded")
            except Exception as e:
                print("⚠️ scaler load skipped:", e)

        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                print("✓ scheduler state loaded")
            except Exception as e:
                print("⚠️ scheduler load failed, will fall back to last_epoch/global_step. err:", e)

        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", start_epoch * len(dataloader))
        scheduler.last_epoch = global_step - 1
        print(f"Resuming from epoch={start_epoch}, global_step={global_step}")

        rng = ckpt.get("rng_state")
        if rng:
            try:
                torch.set_rng_state(rng["cpu"])
                if torch.cuda.is_available() and rng.get("cuda"):
                    torch.cuda.set_rng_state_all(rng["cuda"])
                print("✓ RNG state restored")
            except Exception as e:
                print("⚠️ RNG restore skipped:", e)

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
            probs = samples.get("probs")

            is_rollback = samples.get("is_rollback")
            is_combination = samples.get("is_combination")
            rollback_input_ids = samples.get("rollback_input_ids")
            rollback_attention_mask = samples.get("rollback_attention_mask")
            rollback_images = samples.get("rollback_images")
            combination_input_ids = samples.get("combination_input_ids")
            combination_attention_mask = samples.get("combination_attention_mask")

            if cfg["dataset"] == "200k":
                try:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
                             "combination_attention_mask": combination_attention_mask,
                             "probs": probs
                             }
                        )

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                        global_step += 1
                        losses.update(loss.detach().cpu().item())

                except Exception as e:
                    print("❌ error occurred:", e)
                    raise

            if idx % 1000 == 0:
                print(
                    "Train Epoch ({0}): [{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                        epoch, idx, loss=losses
                    )
                )
        if epoch % cfg["validation_frequency"] == 0:
            print(
                "Train Epoch: [{0}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    loss=losses,
                )
            )
            model_path = cfg["model_path"]
            save_model(
                f"{model_path}/epoch{epoch}.pth",
                cur_epoch=epoch,
                model_to_save=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                extra={"stage": kwargs["stage"], "dataset": cfg["dataset"]},
                save_rng_state=True,
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    setup_seed(42)

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    sys.path.append(config["root_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["dataset"] == "200k":
        train(config, stage="retrospective", device=device)
