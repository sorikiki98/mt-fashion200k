import csv
import json
import os
from os import listdir
from os.path import isfile
from os.path import join


def extract_siblings_per_category(words_dict, category):
    filtered_words_dict = {parent: value[category] for parent, value in words_dict.items() if category in value}
    return filtered_words_dict


def extract_image_names_and_captions(path, split="test"):
    label_path = path + "/labels/"
    label_files = [
        f for f in listdir(label_path) if isfile(join(label_path, f))
    ]
    label_files = [f for f in label_files if split in f]
    image_names = []
    captions = []

    for filename in label_files:
        print("read " + filename)
        with open(label_path + "/" + filename, "rt", encoding="UTF8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.split("	")
            image_name, caption = line[0], line[2]
            if image_name not in image_names:
                image_names.append(image_name)
                captions.append(caption)

    return image_names, captions


def export_siblings_per_category(words_dict, category):
    csv_file_name = f"parent2different_words_{category}.csv"

    with open(csv_file_name, mode='w', newline='', encoding='utf-8-sig') as f:

        writer = csv.writer(f)
        max_values_len = max(len(v) for v in words_dict.values())

        header = ['key'] + [f'value{i + 1}' for i in range(max_values_len)]
        writer.writerow(header)

        for key, values in words_dict.items():
            if len(values) >= 1:
                row = [key] + values
                row += [''] * (max_values_len - len(values))
                writer.writerow(row)


def export_transactions_json(transactions, split="train", name="convergence"):
    transactions_turn3 = [{"id": t["turn-1"]["source_img_id"],
                           "n_turns": t["n_turns"],
                           "turn-1": t["turn-1"],
                           "turn-2": t["turn-2"],
                           "turn-3": t["turn-3"],
                           "probs": t["last_turn_probs"]} for t in transactions if t["n_turns"] == 3]
    transactions_turn4 = [{"id": t["turn-1"]["source_img_id"],
                           "n_turns": t["n_turns"],
                           "turn-1": t["turn-1"],
                           "turn-2": t["turn-2"],
                           "turn-3": t["turn-3"],
                           "turn-4": t["turn-4"],
                           "probs": t["last_turn_probs"]} for t in transactions if t["n_turns"] == 4]
    transactions_turn5 = [{"id": t["turn-1"]["source_img_id"],
                           "n_turns": t["n_turns"],
                           "turn-1": t["turn-1"],
                           "turn-2": t["turn-2"],
                           "turn-3": t["turn-3"],
                           "turn-4": t["turn-4"],
                           "turn-5": t["turn-5"],
                           "probs": t["last_turn_probs"]} for t in transactions if t["n_turns"] == 5]

    transactions = transactions_turn3 + transactions_turn4 + transactions_turn5

    with open(f"{split}_{name}.json", "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=4, ensure_ascii=False)
