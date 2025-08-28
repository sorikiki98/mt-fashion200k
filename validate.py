from operator import itemgetter
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import collate_fn
import random
from pathlib import Path
from PIL import Image


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


def visualize_result_for_single_transaction(model, relative_val_dataset, index_features, sample_idx):
    transaction = relative_val_dataset[sample_idx]

    images = transaction["pil_images"]  # list (6, 3, 224, 224)
    images = [img.unsqueeze(0) for img in images]
    mod_input_ids = transaction["mod_input_ids"]  # list (5, 40)
    mod_input_ids = [ids.unsqueeze(0) for ids in mod_input_ids]
    mod_attention_mask = transaction["mod_attention_mask"]  # list (5, 40)
    mod_attention_mask = [mask.unsqueeze(0) for mask in mod_attention_mask]
    cap_input_ids = transaction["cap_input_ids"]  # list (5, 12)
    cap_input_ids = [ids.unsqueeze(0) for ids in cap_input_ids]
    cap_attention_mask = transaction["cap_attention_mask"]  # list (5, 12)
    cap_attention_mask = [mask.unsqueeze(0) for mask in cap_attention_mask]
    n_turns = torch.tensor([transaction["n_turns"]]) # list 5
    image_paths = transaction["image_paths"]  # list

    first_target_name = str(os.path.dirname(image_paths[1]))
    second_target_name = str(os.path.dirname(image_paths[2]))
    if n_turns == 3:
        last_target_name = str(os.path.dirname(image_paths[3]))
    elif n_turns == 4:
        last_target_name = str(os.path.dirname(image_paths[4]))
    else:
        last_target_name = str(os.path.dirname(image_paths[5]))

    with torch.amp.autocast("cuda"):
        first_distance, second_distance, last_distance = model.inference({
            "target_feats": index_features,
            "n_turns": n_turns,
            "images": images,
            "mod_input_ids": mod_input_ids,
            "mod_attention_mask": mod_attention_mask,
            "cap_input_ids": cap_input_ids,
            "cap_attention_mask": cap_attention_mask,
        })

    return first_distance, second_distance, last_distance, first_target_name, second_target_name, last_target_name


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

        is_rollback = samples.get("is_rollback")
        is_combination = samples.get("is_combination")

        rollback_input_ids = samples.get("rollback_input_ids")
        rollback_attention_mask = samples.get("rollback_attention_mask")
        rollback_images = samples.get("rollback_images")
        combination_input_ids = samples.get("combination_input_ids")
        combination_attention_mask = samples.get("combination_attention_mask")

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
                "cap_attention_mask": cap_attention_mask,
                # "is_rollback": is_rollback,
                # "is_combination": is_combination,
                # "rollback_input_ids": rollback_input_ids,
                # "rollback_attention_mask": rollback_attention_mask,
                # "rollback_images": rollback_images,
                # "combination_input_ids": combination_input_ids,
                # "combination_attention_mask": combination_attention_mask
            })
            first_distance.append(batch_first_distance)
            second_distance.append(batch_second_distance)
            last_distance.append(batch_last_distance)
    first_distance = torch.vstack(first_distance).cpu()
    second_distance = torch.vstack(second_distance).cpu()
    last_distance = torch.vstack(last_distance).cpu()
    return (first_distance, second_distance, last_distance, first_target_names_list, second_target_names_list,
            last_target_names_list)


def save_top_k_retrieval_results(
        first_distance,
        second_distance,
        last_distance,
        first_target_name,
        second_target_name,
        last_target_name,
        index_names,
        output_dir="visualization_results",
        top_k=20
):
    """
    Save top-k retrieved images for each turn based on distances.

    Args:
        first_distance: Distance tensor for first turn
        second_distance: Distance tensor for second turn
        last_distance: Distance tensor for last turn
        first_target_name: Target directory name for first turn
        second_target_name: Target directory name for second turn
        last_target_name: Target directory name for last turn
        index_names: List of image paths in the index
        output_dir: Directory to save results
        top_k: Number of top results to save
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_turn_results(distances, turn_name, target_name):
        """Helper function to save top-k images for a single turn."""
        # Convert to CPU and squeeze batch dimension
        distances = distances.cpu().squeeze()

        # Get indices of top-k smallest distances
        sorted_indices = torch.argsort(distances)[:top_k]

        # Create turn-specific directory
        turn_dir = output_dir / f"{turn_name}_top{top_k}"
        turn_dir.mkdir(exist_ok=True)

        saved_info = []

        for rank, idx in enumerate(sorted_indices):
            idx = idx.item()
            image_path = index_names[idx]
            distance = distances[idx].item()

            # Check if this is the correct match
            is_correct = (image_path == target_name)

            try:
                # Load and save image
                image_path = image_path + "/" + os.path.basename(image_path) + "_0.jpeg"
                img = Image.open(image_path)

                # Create descriptive filename
                correct_indicator = "CORRECT_" if is_correct else ""
                filename = f"{correct_indicator}rank{rank + 1:02d}_dist{distance:.4f}_{os.path.basename(image_path)}"
                save_path = turn_dir / filename

                img.save(save_path)

                saved_info.append({
                    "rank": rank + 1,
                    "distance": distance,
                    "path": image_path,
                    "is_correct": is_correct,
                    "saved_as": str(save_path)
                })

                print(
                    f"[{turn_name}] Rank {rank + 1:2d}: {'✓' if is_correct else ' '} dist={distance:.4f} - {os.path.basename(image_path)}")

            except Exception as e:
                print(f"Error loading/saving image {image_path}: {e}")
                continue

        return saved_info

    # Process each turn
    print(f"\n{'=' * 60}")
    print(f"Saving Top-{top_k} Retrieval Results")
    print(f"{'=' * 60}\n")

    results = {}

    print(f"Turn 1 (Target: {first_target_name})")
    print("-" * 40)
    results["turn1"] = save_turn_results(first_distance, "turn1", first_target_name)

    print(f"\nTurn 2 (Target: {second_target_name})")
    print("-" * 40)
    results["turn2"] = save_turn_results(second_distance, "turn2", second_target_name)

    if last_distance is not None:
        print(f"\nLast Turn (Target: {last_target_name})")
        print("-" * 40)
        results["last"] = save_turn_results(last_distance, "last", last_target_name)

    # Save summary file
    summary_path = output_dir / "retrieval_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RETRIEVAL RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Top-K Retrieved: {top_k}\n\n")

        f.write("TARGET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Turn 1: {first_target_name}\n")
        f.write(f"Turn 2: {second_target_name}\n")
        if last_distance is not None:
            f.write(f"Last Turn: {last_target_name}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("RETRIEVAL PERFORMANCE:\n")
        f.write("-" * 40 + "\n")

        for turn, turn_results in results.items():
            correct_ranks = [r["rank"] for r in turn_results if r["is_correct"]]
            if correct_ranks:
                f.write(f"{turn}: Target found at rank(s) {correct_ranks}\n")
            else:
                f.write(f"{turn}: Target NOT found in top-{top_k}\n")

    print(f"\n{'=' * 60}")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"{'=' * 60}\n")

    return results
