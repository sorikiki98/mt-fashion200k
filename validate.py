from operator import itemgetter
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import collate_fn, device


def compute_blip_compose_multi(relative_val_dataset, blip_model, index_feats,
                               index_names, txt_processors, dataset_name):
    first_pred_sim, second_pred_sim, last_pred_sim, first_target_names, second_target_names, last_target_names, = (
        generate_blip_compose_multi(blip_model, relative_val_dataset, index_feats, txt_processors))

    first_recall_at1, first_recall_at5, first_recall_at10, first_recall_at20 = calculate_recall(first_pred_sim,
                                                                                                index_names,
                                                                                                first_target_names)

    second_recall_at1, second_recall_at5, second_recall_at10, second_recall_at20 = calculate_recall(second_pred_sim,
                                                                                                    index_names,
                                                                                                    second_target_names)
    last_recall_at1, last_recall_at5, last_recall_at10, last_recall_at20 = calculate_recall(last_pred_sim,
                                                                                            index_names,
                                                                                            last_target_names)
    return {
        "Turn 1": (first_recall_at1, first_recall_at5, first_recall_at10, first_recall_at20),
        "Turn 2": (second_recall_at1, second_recall_at5, second_recall_at10, second_recall_at20),
        "Last": (last_recall_at1, last_recall_at5, last_recall_at10, last_recall_at20),
    }


def calculate_recall(pred_sim, index_names, target_names):
    last_distances = 1 - pred_sim
    sorted_indices = torch.argsort(last_distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))

    sums = labels.any(dim=-1).float()

    if not torch.equal(sums.int(), torch.ones(len(target_names)).int()):
        for i, s in enumerate(sums):
            if s != 1:
                print(f"[ERROR] Target '{target_names[i]}' matched {s.item()} times in index_names.")
        raise AssertionError("Some target_names did not match exactly one entry in index_names.")

    recall_at1 = labels[:, :1].any(dim=1).float().mean().item() * 100
    recall_at5 = labels[:, :5].any(dim=1).float().mean().item() * 100
    recall_at10 = labels[:, :10].any(dim=1).float().mean().item() * 100
    recall_at20 = labels[:, :20].any(dim=1).float().mean().item() * 100
    return recall_at1, recall_at5, recall_at10, recall_at20


def generate_blip_compose_multi(blip_model, relative_val_dataset,
                                index_features, txt_processors):
    relative_val_loader = DataLoader(dataset=relative_val_dataset,
                                     batch_size=16,
                                     num_workers=8,
                                     pin_memory=True,
                                     collate_fn=collate_fn,
                                     shuffle=False,
                                     persistent_workers=True, )

    first_target_names_list = []
    second_target_names_list = []
    last_target_names_list = []
    first_distance = []
    second_distance = []
    last_distance = []

    for n_turns, ref_img, ref_name, tar_imgs, tar_names, mods in tqdm(relative_val_loader, desc="Val"):
        mods = list(zip(*mods))
        mod1_inputs = [txt_processors["eval"](m[0]) for m in mods]
        mod2_inputs = [txt_processors["eval"](m[1]) for m in mods]
        mod3_inputs = [txt_processors["eval"](m[2]) for m in mods]
        mod4_inputs = [txt_processors["eval"](m[3]) for m in mods]
        mod5_inputs = [txt_processors["eval"](m[4]) for m in mods]

        ref_img = ref_img.to(device, non_blocking=True).half()
        tar1_img = tar_imgs[:, 0].to(device, non_blocking=True)  # torch.size([32, 3, 225, 225])
        tar2_img = tar_imgs[:, 1].to(device, non_blocking=True)
        tar3_img = tar_imgs[:, 2].to(device, non_blocking=True)
        tar4_img = tar_imgs[:, 3].to(device, non_blocking=True)
        tar5_img = tar_imgs[:, 4].to(device, non_blocking=True)

        tar_names = list(zip(*tar_names))
        tar1_name = [name[0] for name in tar_names]
        tar2_name = [name[1] for name in tar_names]
        tar3_name = [name[2] for name in tar_names]
        tar4_name = [name[3] for name in tar_names]
        tar5_name = [name[4] for name in tar_names]

        for i in range(len(n_turns)):
            turn = n_turns[i]
            first_target_names_list.append(str(os.path.dirname(tar1_name[i])))
            second_target_names_list.append(str(os.path.dirname(tar2_name[i])))
            if turn == 3:
                last_target_names_list.append(str(os.path.dirname(tar3_name[i])))
            elif turn == 4:
                last_target_names_list.append(str(os.path.dirname(tar4_name[i])))
            elif turn == 5:
                last_target_names_list.append(str(os.path.dirname(tar5_name[i])))

        with torch.amp.autocast("cuda"):
            batch_first_distance, batch_second_distance, batch_last_distance = blip_model.inference({
                "target_feats": index_features,
                "n_turns": n_turns,
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
                "tar5_img": tar5_img
            })
            first_distance.append(batch_first_distance)
            second_distance.append(batch_second_distance)
            last_distance.append(batch_last_distance)
    first_distance = torch.vstack(first_distance).cpu()
    second_distance = torch.vstack(second_distance).cpu()
    last_distance = torch.vstack(last_distance).cpu()
    return (first_distance, second_distance, last_distance, first_target_names_list, second_target_names_list,
            last_target_names_list)
