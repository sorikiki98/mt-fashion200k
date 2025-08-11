import random
from fashion200k import Fashion200k
from fashion_words import colors, items, material, pattern, silhouettes, structures, details, style, functionalities
from fashion_prompts import replacement, addition


def add_transaction_types(results):
    results["rollback"] = False
    results["combination"] = False
    return results


class Fashion200kConvergence(Fashion200k):
    def __init__(self, path, seed=42, split="train"):
        super().__init__(path, seed, split)
        self.transactions = list()
        for idx in range(len(self.imgs)):
            mod_type = []
            add_type = []
            result1 = self.turn1_sample_(idx, mod_type.copy(), 1)
            if result1 is not None:
                result2 = self.turn2_sample_(result1["target_img_id"], result1["mod_type"].copy(), add_type.copy(),
                                             2)
                if result2 is not None:
                    result3 = self.turn3_sample_(result2["target_img_id"], result2["mod_type"].copy(),
                                                 result2["add_type"].copy(), 3)
                    if result3 is not None:
                        result4 = self.turn3_sample_(result3["target_img_id"], result3["mod_type"].copy(),
                                                     result3["add_type"].copy(), 4)
                        if result4 is not None:
                            result5 = self.turn3_sample_(result4["target_img_id"], result4["mod_type"].copy(),
                                                         result4["add_type"].copy(), 5)
                            if result5 is not None:
                                results = {"n_turns": 5, "turn-1": result1, "turn-2": result2, "turn-3": result3,
                                           "turn-4": result4, "turn-5": result5}
                                results = add_transaction_types(results)
                                self.transactions.append(results)
                            else:
                                results = {"n_turns": 4, "turn-1": result1, "turn-2": result2, "turn-3": result3,
                                           "turn-4": result4}
                                results = add_transaction_types(results)
                                self.transactions.append(results)
                        else:
                            results = {"n_turns": 3, "turn-1": result1, "turn-2": result2, "turn-3": result3}
                            results = add_transaction_types(results)
                            self.transactions.append(results)

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, idx):
        return self.transactions[idx]

    def turn1_sample_(self, idx, mod_type, n_turn):
        img = self.imgs[idx]
        for p in img["parent_captions"]:
            child = set(img["captions"][0].split())
            parent = set(p.split())
            source_word = list(child - parent)[0]
            if (all(not m.startswith("color") for m in mod_type)
                    and source_word in colors
                    and p in self.parent2different_colors
                    and len(self.parent2different_colors[p]) >= 2):
                target_tuples = [t for t in self.parent2different_colors[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_colors[p] if t[0] == source_word][0]
                mod_type.append(f"color-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, [], n_turn)
            elif (all(not m.startswith("item") for m in mod_type)
                  and source_word in items
                  and p in self.parent2different_items
                  and len(self.parent2different_items[p]) >= 2):
                target_tuples = [t for t in self.parent2different_items[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_items[p] if t[0] == source_word][0]
                mod_type.append(f"item-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, [], n_turn)
            elif (all(not m.startswith("pattern") for m in mod_type)
                  and source_word in pattern
                  and p in self.parent2different_pattern
                  and len(self.parent2different_pattern[p]) >= 2):
                target_tuples = [t for t in self.parent2different_pattern[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_pattern[p] if t[0] == source_word][0]
                mod_type.append(f"pattern-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, [], n_turn)
        # turn 1에서는 add_new_attributes 호출 X => 눈에 띄는 변화를 적용 하기 위함
        return

    def turn2_sample_(self, idx, mod_type, add_type, n_turn):
        img = self.imgs[idx]
        for p in img["parent_captions"]:
            child = set(img["captions"][0].split())
            parent = set(p.split())
            source_word = list(child - parent)[0]
            if (all(not m.startswith("structure") for m in mod_type)
                    and source_word in structures
                    and p in self.parent2different_structures
                    and len(self.parent2different_structures[p]) >= 2):
                target_tuples = [t for t in self.parent2different_structures[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_structures[p] if t[0] == source_word][0]
                mod_type.append(f"structure-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif (all(not m.startswith("silhouette") for m in mod_type)
                  and source_word in silhouettes
                  and p in self.parent2different_silhouettes
                  and len(self.parent2different_silhouettes[p]) >= 2):
                target_tuples = [t for t in self.parent2different_silhouettes[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_silhouettes[p] if t[0] == source_word][0]
                mod_type.append(f"silhouette-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
        if (any(attr in colors for attr in img["captions"][0].split()) and "color" not in mod_type
                or any(attr in items for attr in img["captions"][0].split()) and "item" not in mod_type
                or any(attr in pattern for attr in img["captions"][0].split()) and "pattern" not in mod_type):
            return self.turn1_sample_(idx, mod_type, n_turn)
        return self.add_new_attribute(idx, mod_type, add_type, n_turn)

    def turn3_sample_(self, idx, mod_type, add_type, n_turn):
        img = self.imgs[idx]
        for p in img["parent_captions"]:
            child = set(img["captions"][0].split())
            parent = set(p.split())
            source_word = list(child - parent)[0]
            if (all(not m.startswith("detail") for m in mod_type)
                    and source_word in details
                    and p in self.parent2different_details
                    and len(self.parent2different_details[p]) >= 2):
                target_tuples = [t for t in self.parent2different_details[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_details[p] if t[0] == source_word][0]
                mod_type.append(f"detail-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif (all(not m.startswith("material") for m in mod_type)
                  and source_word in material
                  and p in self.parent2different_material
                  and len(self.parent2different_material[p]) >= 2):
                target_tuples = [t for t in self.parent2different_material[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_material[p] if t[0] == source_word][0]
                mod_type.append(f"material-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif (all(not m.startswith("style") for m in mod_type)
                  and source_word in style
                  and p in self.parent2different_style
                  and len(self.parent2different_style[p]) >= 2):
                target_tuples = [t for t in self.parent2different_style[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_style[p] if t[0] == source_word][0]
                mod_type.append(f"style-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif (all(not m.startswith("functionality") for m in mod_type)
                  and source_word in functionalities
                  and p in self.parent2different_functionalities
                  and len(self.parent2different_functionalities[p]) >= 2):
                target_tuples = [t for t in self.parent2different_functionalities[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_functionalities[p] if t[0] == source_word][0]
                mod_type.append(f"functionality-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
        if (any(attr in structures for attr in img["captions"][0].split()) and "structure" not in mod_type
                or any(attr in silhouettes for attr in img["captions"][0].split()) and "silhouette" not in mod_type):
            return self.turn2_sample_(idx, mod_type, add_type, n_turn)
        return self.add_new_attribute(idx, mod_type, add_type, n_turn)

    def add_new_attribute(self, idx, mod_type, add_type, n_turn):
        img = self.imgs[idx]
        for c in img["captions"]:
            if c in self.parent2different_pattern:
                target_tuple = random.choice(self.parent2different_pattern[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"pattern-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif c in self.parent2different_details:
                target_tuple = random.choice(self.parent2different_details[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"detail-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif c in self.parent2different_items:
                target_tuple = random.choice(self.parent2different_items[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"item-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif c in self.parent2different_style:
                target_tuple = random.choice(self.parent2different_style[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"style-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif c in self.parent2different_silhouettes:
                target_tuple = random.choice(self.parent2different_silhouettes[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"silhouette-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif c in self.parent2different_structures:
                target_tuple = random.choice(self.parent2different_structures[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"structure-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif c in self.parent2different_colors:
                target_tuple = random.choice(self.parent2different_colors[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"color-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif c in self.parent2different_material:
                target_tuple = random.choice(self.parent2different_material[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"material-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif c in self.parent2different_functionalities:
                target_tuple = random.choice(self.parent2different_functionalities[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                add_type.append(f"functionality-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
        return

    def add_single_turn(self, idx, source_tuple, target_tuple, mod_type, add_type, n_turn):
        source_word, source_caption_id = source_tuple
        target_word, target_caption_id = target_tuple
        source_caption = self.get_caption(source_caption_id)
        target_caption = self.get_caption(target_caption_id)
        target_idx = random.choice(self.caption2imgids[target_caption])
        if source_word == "":
            template = random.choice(addition)
            mod_str = f"Turn {n_turn}: " + template.format(new_attr=target_word)
        else:
            template = random.choice(replacement)
            mod_str = f"Turn {n_turn}: " + template.format(old_attr=source_word, new_attr=target_word)
        return {
            "source_img_id": idx,
            "target_img_id": target_idx,
            "source_word": source_word,
            "target_word": target_word,
            "source_caption": source_caption,
            "target_caption": target_caption,
            "source_img_path": self.imgs[idx]["file_path"],
            "target_img_path": self.imgs[target_idx]["file_path"],
            "mod_str": mod_str,
            "mod_type": mod_type,
            "add_type": add_type
        }
