import random
from fashion200k import Fashion200k
from fashion_words import colors, items, material, pattern, silhouettes, structures, details, style, functionalities
from fashion_prompts import combination, replacement, addition


class Fashion200kCombination(Fashion200k):
    def __init__(self, path, seed=155, split="train"):
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
                            result5 = self.combination_(result1, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                        5, 1) \
                                if self.combination_(result1, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                     5, 1) \
                                else self.combination_(result2, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                       5, 2) \
                                if self.combination_(result2, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                     5, 2) \
                                else self.combination_(result3, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                       5, 3) \
                                if self.combination_(result3, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                     5, 3) \
                                else self.combination_(result4, result4["mod_type"].copy(), result4["add_type"].copy(),
                                                       5, 4)

                            if result5 is not None:
                                self.transactions.append(
                                    {"n_turns": 5, "turn-1": result1, "turn-2": result2, "turn-3": result3,
                                     "turn-4": result4, "turn-5": result5})
                            else:
                                result4 = self.combination_(result1, result3["mod_type"].copy(),
                                                            result3["add_type"].copy(),
                                                            4, 1) \
                                    if self.combination_(result1, result3["mod_type"].copy(),
                                                         result3["add_type"].copy(),
                                                         4, 1) \
                                    else self.combination_(result2, result3["mod_type"].copy(),
                                                           result3["add_type"].copy(),
                                                           4, 2) \
                                    if self.combination_(result2, result3["mod_type"].copy(),
                                                         result3["add_type"].copy(),
                                                         4, 2) \
                                    else self.combination_(result3, result3["mod_type"].copy(),
                                                           result3["add_type"].copy(),
                                                           4, 3)
                                if result4 is not None:
                                    self.transactions.append(
                                        {"n_turns": 4, "turn-1": result1, "turn-2": result2, "turn-3": result3,
                                         "turn-4": result4})
                                else:
                                    result3 = self.combination_(result1, result2["mod_type"].copy(),
                                                                result2["add_type"].copy(),
                                                                3, 1) \
                                        if self.combination_(result1, result2["mod_type"].copy(),
                                                             result2["add_type"].copy(),
                                                             3, 1) \
                                        else self.combination_(result2, result2["mod_type"].copy(),
                                                               result2["add_type"].copy(),
                                                               3, 2)
                                    if result3 is not None:
                                        self.transactions.append(
                                            {"n_turns": 3, "turn-1": result1, "turn-2": result2,
                                             "turn-3": result3})
                        else:
                            result4 = self.combination_(result1, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                        4, 1) \
                                if self.combination_(result1, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                     4, 1) \
                                else self.combination_(result2, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                       4, 2) \
                                if self.combination_(result2, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                     4, 2) \
                                else self.combination_(result3, result3["mod_type"].copy(), result3["add_type"].copy(),
                                                       4, 3)
                            if result4 is not None:
                                self.transactions.append(
                                    {"n_turns": 4, "turn-1": result1, "turn-2": result2, "turn-3": result3,
                                     "turn-4": result4})
                    else:
                        result3 = self.combination_(result1, result2["mod_type"].copy(),
                                                    result2["add_type"].copy(),
                                                    3, 1) \
                            if self.combination_(result1, result2["mod_type"].copy(),
                                                 result2["add_type"].copy(),
                                                 3, 1) \
                            else self.combination_(result2, result2["mod_type"].copy(),
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
                    and (p in self.parent2different_items or p in self.parent2different_pattern or
                         p in self.parent2different_colors)):
                if p in self.parent2different_items:
                    target_tuples = [t for t in self.parent2different_items[p]]
                elif p in self.parent2different_pattern:
                    target_tuples = [t for t in self.parent2different_pattern[p]]
                else:
                    target_tuples = [t for t in self.parent2different_colors[p]]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_colors[p] if t[0] == source_word][0]
                mod_type.append(f"color-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, [], n_turn)
            elif (all(not m.startswith("item") for m in mod_type)
                  and source_word in items
                  and (p in self.parent2different_pattern or p in self.parent2different_colors or
                       p in self.parent2different_items)):
                if p in self.parent2different_pattern:
                    target_tuples = [t for t in self.parent2different_pattern[p]]
                elif p in self.parent2different_colors:
                    target_tuples = [t for t in self.parent2different_colors[p]]
                else:
                    target_tuples = [t for t in self.parent2different_items[p]]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_items[p] if t[0] == source_word][0]
                mod_type.append(f"item-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, [], n_turn)
            elif (all(not m.startswith("pattern") for m in mod_type)
                  and source_word in pattern
                  and (p in self.parent2different_colors or p in self.parent2different_items or
                       p in self.parent2different_pattern)):
                if p in self.parent2different_colors:
                    target_tuples = [t for t in self.parent2different_colors[p]]
                elif p in self.parent2different_items:
                    target_tuples = [t for t in self.parent2different_items[p]]
                else:
                    target_tuples = [t for t in self.parent2different_pattern[p]]
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
                    and (p in self.parent2different_silhouettes or p in self.parent2different_structures)):
                if p in self.parent2different_silhouettes:
                    target_tuples = [t for t in self.parent2different_silhouettes[p]]
                else:
                    target_tuples = [t for t in self.parent2different_structures[p]]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_structures[p] if t[0] == source_word][0]
                mod_type.append(f"structure-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif (all(not m.startswith("silhouette") for m in mod_type)
                  and source_word in silhouettes
                  and (p in self.parent2different_structures or p in self.parent2different_silhouettes)):
                if p in self.parent2different_structures:
                    target_tuples = [t for t in self.parent2different_structures[p]]
                else:
                    target_tuples = [t for t in self.parent2different_silhouettes[p]]
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
                    and (p in self.parent2different_material or p in self.parent2different_style or
                         p in self.parent2different_functionalities or p in self.parent2different_details)):
                if p in self.parent2different_details:
                    target_tuples = [t for t in self.parent2different_details[p]]
                elif p in self.parent2different_material:
                    target_tuples = [t for t in self.parent2different_material[p]]
                elif p in self.parent2different_style:
                    target_tuples = [t for t in self.parent2different_style[p]]
                else:
                    target_tuples = [t for t in self.parent2different_functionalities[p]]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_details[p] if t[0] == source_word][0]
                mod_type.append(f"detail-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif (all(not m.startswith("material") for m in mod_type)
                  and source_word in material
                  and (p in self.parent2different_style or p in self.parent2different_functionalities or
                       p in self.parent2different_details)):
                if p in self.parent2different_style:
                    target_tuples = [t for t in self.parent2different_style[p]]
                elif p in self.parent2different_functionalities:
                    target_tuples = [t for t in self.parent2different_functionalities[p]]
                else:
                    target_tuples = [t for t in self.parent2different_details[p]]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_material[p] if t[0] == source_word][0]
                mod_type.append(f"material-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif (all(not m.startswith("style") for m in mod_type)
                  and source_word in style
                  and (p in self.parent2different_functionalities or p in self.parent2different_details or
                       p in self.parent2different_material)):
                if p in self.parent2different_functionalities:
                    target_tuples = [t for t in self.parent2different_functionalities[p]]
                elif p in self.parent2different_details:
                    target_tuples = [t for t in self.parent2different_details[p]]
                else:
                    target_tuples = [t for t in self.parent2different_material[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_style[p] if t[0] == source_word][0]
                mod_type.append(f"style-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
            elif (all(not m.startswith("functionality") for m in mod_type)
                  and source_word in functionalities
                  and (p in self.parent2different_details or p in self.parent2different_material or
                       p in self.parent2different_style)):
                if p in self.parent2different_details:
                    target_tuples = [t for t in self.parent2different_details[p]]
                elif p in self.parent2different_material:
                    target_tuples = [t for t in self.parent2different_material[p]]
                else:
                    target_tuples = [t for t in self.parent2different_style[p]]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_functionalities[p] if t[0] == source_word][0]
                mod_type.append(f"functionality-{n_turn}")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type, add_type, n_turn)
        if (any(attr in structures for attr in img["captions"][0].split()) and "structure" not in mod_type
                or any(attr in silhouettes for attr in img["captions"][0].split()) and "silhouette" not in mod_type):
            return self.turn2_sample_(idx, mod_type, add_type, n_turn)
        return

    def combination_(self, c_result, mod_type, add_type, n_turn, c_turn):
        src1_idx = c_result["source_img_id"]
        src1_word = c_result["source_word"]
        src1_img = self.imgs[src1_idx]
        src1_caption = src1_img["captions"][0]

        src2_idx = c_result["target_img_id"]
        src2_word = c_result["target_word"]
        src2_img = self.imgs[src2_idx]
        src2_caption = src2_img["captions"][0]

        src1_caption_list = src1_caption.split()
        parent_caption_candidates = []
        for idx in range(len(src1_caption_list) + 1):
            temp = src1_caption_list[:]
            temp.insert(idx, src2_word)
            parent_caption_candidates.append(temp)
        parent_caption_candidates = [" ".join(caption_list) for caption_list in parent_caption_candidates]

        if parent_caption := next((c for c in parent_caption_candidates if c in self.parent2different_colors), None):
            target_tuple = random.choice(self.parent2different_colors[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"color-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)
        elif parent_caption := next((c for c in parent_caption_candidates if c in self.parent2different_items), None):
            target_tuple = random.choice(self.parent2different_items[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"item-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)
        elif parent_caption := next((c for c in parent_caption_candidates if c in self.parent2different_pattern), None):
            target_tuple = random.choice(self.parent2different_pattern[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"pattern-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)
        elif parent_caption := next((c for c in parent_caption_candidates if c in self.parent2different_silhouettes),
                                    None):
            target_tuple = random.choice(self.parent2different_silhouettes[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"silhouette-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)
        elif parent_caption := next((c for c in parent_caption_candidates if c in self.parent2different_structures),
                                    None):
            target_tuple = random.choice(self.parent2different_structures[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"structure-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)
        elif parent_caption := next((c for c in parent_caption_candidates if c in self.parent2different_details),
                                    None):
            target_tuple = random.choice(self.parent2different_details[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"detail-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)
        elif parent_caption := next((c for c in parent_caption_candidates if c in self.parent2different_material),
                                    None):
            target_tuple = random.choice(self.parent2different_material[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"material-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)

        elif parent_caption := next((c for c in parent_caption_candidates if c in self.parent2different_style),
                                    None):
            target_tuple = random.choice(self.parent2different_style[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"style-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)

        elif parent_caption := next(
                (c for c in parent_caption_candidates if c in self.parent2different_functionalities), None):
            target_tuple = random.choice(self.parent2different_functionalities[parent_caption])
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            add_type.append(f"functionality-{n_turn}")
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)
        if caption := next(
                (c for c in parent_caption_candidates if c in self.caption2imgids), None):
            target_caption_id = self.caption2id[caption]
            target_tuple = ("", target_caption_id)
            src1_caption_id = self.caption2id[src1_caption]
            src1_tuple = (src1_word, src1_caption_id)
            src2_caption_id = self.caption2id[src2_caption]
            src2_tuple = (src2_word, src2_caption_id)
            return self.add_single_turn_combination(src1_idx, src2_idx, src1_tuple, src2_tuple, target_tuple,
                                                    mod_type, add_type, c_turn, n_turn)

    def add_single_turn_combination(self, idx1, idx2, src1_tuple, src2_tuple, target_tuple, mod_type, add_type, c_turn,
                                    n_turn):
        src1_word, src1_caption_id = src1_tuple
        src2_word, src2_caption_id = src2_tuple
        target_word, target_caption_id = target_tuple
        target_caption = self.get_caption(target_caption_id)
        target_idx = random.choice(self.caption2imgids[target_caption])
        template = random.choice(combination)
        if c_turn == 1:
            mod_str = f"Turn {n_turn}: " + template.format(n_turn=1, old_attr1=src1_word, old_attr2=src2_word,
                                                           new_attr=target_word)
        elif c_turn == 2:
            mod_str = f"Turn {n_turn}: " + template.format(n_turn=2, old_attr1=src1_word, old_attr2=src2_word,
                                                           new_attr=target_word)
        else:
            mod_str = f"Turn {n_turn}: " + template.format(n_turn=3, old_attr1=src1_word, old_attr2=src2_word,
                                                           new_attr=target_word)
        return {
            "source_img_id": (idx1, idx2),
            "target_img_id": target_idx,
            "source_word": (src1_word, src2_word),
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
            "source_img_path": self.imgs[idx]["file_path"],
            "target_img_path": self.imgs[target_idx]["file_path"],
            "mod_str": mod_str,
            "mod_type": mod_type,
            "add_type": add_type,
        }
