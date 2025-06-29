import random
from dataset import Fashion200k
from fashion_words import colors, items, material, pattern, silhouettes, structures, details
from fashion_prompts import rollback, replacement, addition


class Fashion200kRollback(Fashion200k):
    def __init__(self, path, seed=71):
        super().__init__(path, seed)
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
                            result5 = self.rollback_(result2, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                     5, 2) \
                                if self.rollback_(result2, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                  5, 2) \
                                else self.rollback_(result3, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                    5, 3) \
                                if self.rollback_(result3, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                  5, 3) \
                                else self.rollback_(result4, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                    5, 4)
                            if result5 is not None:
                                self.transactions.append(
                                    {"n_turns": 5, "turn-1": result1, "turn-2": result2, "turn-3": result3,
                                     "turn-4": result4, "turn-5": result5})
                            else:
                                result4 = self.rollback_(result2, result3["mod_type"].copy(),
                                                         result3["add_type"].copy(),
                                                         4, 2) \
                                    if self.rollback_(result2, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                      4, 2) \
                                    else self.rollback_(result3, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                        4, 3)
                                if result4 is not None:
                                    self.transactions.append(
                                        {"n_turns": 4, "turn-1": result1, "turn-2": result2, "turn-3": result3,
                                         "turn-4": result4})
                                else:
                                    result3 = self.rollback_(result2, result2["mod_type"].copy(),
                                                             result2["add_type"].copy(),
                                                             3, 2)
                                    if result3 is not None:
                                        self.transactions.append(
                                            {"n_turns": 3, "turn-1": result1, "turn-2": result2,
                                             "turn-3": result3})
                        else:
                            result4 = self.rollback_(result2, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                     4, 2) \
                                if self.rollback_(result2, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                  4, 2) \
                                else self.rollback_(result3, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                    4, 3)
                            if result4 is not None:
                                self.transactions.append(
                                    {"n_turns": 4, "turn-1": result1, "turn-2": result2, "turn-3": result3,
                                     "turn-4": result4})
                    else:
                        result3 = self.rollback_(result2, result2["mod_type"].copy(),
                                                 result2["add_type"].copy(),
                                                 3, 2)
                        if result3 is not None:
                            self.transactions.append(
                                {"n_turns": 3, "turn-1": result1, "turn-2": result2,
                                 "turn-3": result3})

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
        return

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
        if (any(attr in structures for attr in img["captions"][0].split()) and "structure" not in mod_type
                or any(attr in silhouettes for attr in img["captions"][0].split()) and "silhouette" not in mod_type):
            return self.turn2_sample_(idx, mod_type, add_type, n_turn)
        return

    def rollback_(self, r_result, mod_type, add_type, r_turn, n_turn):
        r_mod_type = r_result["mod_type"]
        r_add_type = r_result["add_type"]
        max_item = max(r_mod_type + r_add_type, key=lambda t: int(t.split("-")[1]))
        attr_type = max_item.split("-")[0]

        src_idx = r_result["source_img_id"]
        src_word = r_result["source_word"]
        tar_word = r_result["target_word"]

        src_img = self.imgs[src_idx]
        src_caption = src_img["captions"][0]
        parent_caption = src_caption.replace(src_word, "")
        parent_caption = parent_caption.replace("  ", " ").strip()

        if attr_type == "color" and parent_caption in self.parent2different_colors \
                and len(self.parent2different_colors[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_colors[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_colors[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"color-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)
        elif attr_type == "item" and parent_caption in self.parent2different_items \
                and len(self.parent2different_items[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_items[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_items[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"item-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)
        elif attr_type == "material" and parent_caption in self.parent2different_material \
                and len(self.parent2different_material[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_material[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_material[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"material-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)
        elif attr_type == "pattern" and parent_caption in self.parent2different_pattern \
                and len(self.parent2different_pattern[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_pattern[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_pattern[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"pattern-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)
        elif attr_type == "functionality" and parent_caption in self.parent2different_functionalities \
                and len(self.parent2different_functionalities[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_functionalities[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_functionalities[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"functionality-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)
        elif attr_type == "silhouette" and parent_caption in self.parent2different_silhouettes \
                and len(self.parent2different_silhouettes[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_silhouettes[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_silhouettes[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"silhouette-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)
        elif attr_type == "structure" and parent_caption in self.parent2different_structures \
                and len(self.parent2different_structures[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_structures[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_structures[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"structure-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)
        elif attr_type == "detail" and parent_caption in self.parent2different_details \
                and len(self.parent2different_details[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_details[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_details[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"detail-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)
        elif attr_type == "style" and parent_caption in self.parent2different_style \
                and len(self.parent2different_style[parent_caption]) >= 4:
            target_tuples = [t for t in self.parent2different_style[parent_caption]
                             if t[0] != src_word and t[0] != tar_word]
            target_tuple = random.choice(target_tuples)
            source_tuple = [t for t in self.parent2different_style[parent_caption] if t[0] == src_word][0]
            mod_type.append(f"style-{n_turn}")
            return self.add_single_turn_rollback(src_idx, source_tuple, target_tuple, mod_type, add_type, r_turn,
                                                 n_turn)

    def add_single_turn_rollback(self, idx, source_tuple, target_tuple, mod_type, add_type, r_turn, n_turn):
        source_word, source_caption_id = source_tuple
        target_word, target_caption_id = target_tuple
        target_caption = self.get_caption(target_caption_id)
        target_idx = random.choice(self.caption2imgids[target_caption])
        template = random.choice(rollback)
        if r_turn == 2:
            mod_str = f"Turn {n_turn}: " + template.format(n_turn=1, new_attr=target_word)
        elif r_turn == 3:
            mod_str = f"Turn {n_turn}: " + template.format(n_turn=2, new_attr=target_word)
        else:
            mod_str = f"Turn {n_turn}: " + template.format(n_turn=3, new_attr=target_word)
        return {
            "source_img_id": idx,
            "target_img_id": target_idx,
            "source_word": source_word,
            "target_word": target_word,
            "mod_str": mod_str,
            "mod_type": mod_type,
            "add_type": add_type
        }

    def add_single_turn(self, idx, source_tuple, target_tuple, mod_type, add_type, n_turn):
        source_word, source_caption_id = source_tuple
        target_word, target_caption_id = target_tuple
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
            "mod_str": mod_str,
            "mod_type": mod_type,
            "add_type": add_type
        }
