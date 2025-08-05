from operator import itemgetter
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import collate_fn, device


def compute_blip_compose_multi(relative_val_dataset, model, index_feats, index_names):
    first_pred_sim, second_pred_sim, last_pred_sim, first_target_names, second_target_names, last_target_names, = (
        generate_blip_compose_multi(model, relative_val_dataset, index_feats))

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


def generate_blip_compose_multi(model, relative_val_dataset, index_features):
    relative_val_loader = DataLoader(dataset=relative_val_dataset,
                                     batch_size=16,
                                     num_workers=8,
                                     pin_memory=True,
                                     collate_fn=collate_fn,
                                     shuffle=False,
                                     persistent_workers=False, )

    first_target_names_list = []
    second_target_names_list = []
    last_target_names_list = []
    first_distance = []
    second_distance = []
    last_distance = []

    for samples in tqdm(relative_val_loader, desc="Val"):
        images = samples.get("pil_images")
        mod_input_ids = samples.get("mod_input_ids")
        mod_attention_mask = samples.get("mod_attention_mask")
        cap_input_ids = samples.get("cap_input_ids")
        cap_attention_mask = samples.get("cap_attention_mask")
        n_turns = samples.get("n_turns")
        image_paths = samples.get("image_paths")  # (6, B)

        batch_size = images[0].size(0)

        for i in range(batch_size):
            turn = n_turns[i]
            first_target_names_list.append(str(os.path.dirname(image_paths[1][i])))
            second_target_names_list.append(str(os.path.dirname(image_paths[2][i])))
            if turn == 3:
                last_target_names_list.append(str(os.path.dirname(image_paths[3][i])))
            elif turn == 4:
                last_target_names_list.append(str(os.path.dirname(image_paths[4][i])))
            elif turn == 5:
                last_target_names_list.append(str(os.path.dirname(image_paths[5][i])))

        with torch.amp.autocast("cuda"):
            batch_first_distance, batch_second_distance, batch_last_distance = model.inference({
                "target_feats": index_features,
                "n_turns": n_turns,
                "images": images,
                "mod_input_ids": mod_input_ids,
                "mod_attention_mask": mod_attention_mask,
                "cap_input_ids": cap_input_ids,
                "cap_attention_mask": cap_attention_mask
            })
            first_distance.append(batch_first_distance)
            second_distance.append(batch_second_distance)
            last_distance.append(batch_last_distance)
    first_distance = torch.vstack(first_distance).cpu()
    second_distance = torch.vstack(second_distance).cpu()
    last_distance = torch.vstack(last_distance).cpu()
    return (first_distance, second_distance, last_distance, first_target_names_list, second_target_names_list,
            last_target_names_list)
