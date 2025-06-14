import random
from dataset import Fashion200k
from fashion_words import colors, items, material, pattern, silhouettes, structures, details


class Fashion200kConvergence(Fashion200k):
    def __init__(self, path, seed=42):
        super().__init__(path)
        self.transactions = list()
        random.seed(seed)
        for idx in range(len(self.imgs)):
            mod_type = []
            result1 = self.turn1_sample_(idx, mod_type.copy())
            if result1 is not None:
                result2 = self.turn2_sample_(result1["target_img_id"], result1["mod_type"].copy())
                if result2 is not None:
                    result3 = self.turn3_sample_(result2["target_img_id"], result2["mod_type"].copy())
                    if result3 is not None:
                        self.transactions.append({"0": result1, "1": result2, "2": result3})

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, idx):
        return self.transactions[idx]

    def turn1_sample_(self, idx, mod_type):
        img = self.imgs[idx]
        for p in img["parent_captions"]:
            child = set(img["captions"][0].split())
            parent = set(p.split())
            source_word = list(child - parent)[0]
            if ("color" not in mod_type
                    and source_word in colors
                    and p in self.parent2different_colors
                    and len(self.parent2different_colors[p]) >= 2):
                target_tuples = [t for t in self.parent2different_colors[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_colors[p] if t[0] == source_word][0]
                mod_type.append("color")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif ("item" not in mod_type
                  and source_word in items
                  and p in self.parent2different_items
                  and len(self.parent2different_items[p]) >= 2):
                target_tuples = [t for t in self.parent2different_items[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_items[p] if t[0] == source_word][0]
                mod_type.append("item")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif ("pattern" not in mod_type
                  and source_word in pattern
                  and p in self.parent2different_pattern
                  and len(self.parent2different_pattern[p]) >= 2):
                target_tuples = [t for t in self.parent2different_pattern[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_pattern[p] if t[0] == source_word][0]
                mod_type.append("pattern")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
        # turn 1에서는 add_new_attributes 호출 X => 눈에 띄는 변화를 적용 하기 위함
        return

    def turn2_sample_(self, idx, mod_type):
        img = self.imgs[idx]
        for p in img["parent_captions"]:
            child = set(img["captions"][0].split())
            parent = set(p.split())
            source_word = list(child - parent)[0]
            if ("structure" not in mod_type
                    and source_word in structures
                    and p in self.parent2different_structures
                    and len(self.parent2different_structures[p]) >= 2):
                target_tuples = [t for t in self.parent2different_structures[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_structures[p] if t[0] == source_word][0]
                mod_type.append("structure")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif ("silhouette" not in mod_type
                  and source_word in silhouettes
                  and p in self.parent2different_silhouettes
                  and len(self.parent2different_silhouettes[p]) >= 2):
                target_tuples = [t for t in self.parent2different_silhouettes[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_silhouettes[p] if t[0] == source_word][0]
                mod_type.append("silhouette")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
        if (any(attr in colors for attr in img["captions"][0].split()) and "color" not in mod_type
                or any(attr in items for attr in img["captions"][0].split()) and "item" not in mod_type
                or any(attr in pattern for attr in img["captions"][0].split()) and "pattern" not in mod_type):
            return self.turn1_sample_(idx, mod_type)
        return self.add_new_attribute(idx, mod_type)

    def turn3_sample_(self, idx, mod_type):
        img = self.imgs[idx]
        for p in img["parent_captions"]:
            child = set(img["captions"][0].split())
            parent = set(p.split())
            source_word = list(child - parent)[0]
            if (source_word in details
                    and p in self.parent2different_details
                    and len(self.parent2different_details[p]) >= 2):
                target_tuples = [t for t in self.parent2different_details[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_details[p] if t[0] == source_word][0]
                mod_type.append("detail")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif (source_word in material
                  and p in self.parent2different_materials
                  and len(self.parent2different_materials[p]) >= 2):
                target_tuples = [t for t in self.parent2different_materials[p] if t[0] != source_word]
                target_tuple = random.choice(target_tuples)
                source_tuple = [t for t in self.parent2different_materials[p] if t[0] == source_word][0]
                mod_type.append("material")
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
        if (any(attr in structures for attr in img["captions"][0].split()) and "structure" not in mod_type
                or any(attr in silhouettes for attr in img["captions"][0].split()) and "silhouette" not in mod_type):
            return self.turn2_sample_(idx, mod_type)
        return self.add_new_attribute(idx, mod_type)

    def add_new_attribute(self, idx, mod_type):
        img = self.imgs[idx]
        for c in img["captions"]:
            if c in self.parent2different_pattern:
                target_tuple = random.choice(self.parent2different_pattern[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif c in self.parent2different_details:
                target_tuple = random.choice(self.parent2different_details[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif c in self.parent2different_items:
                target_tuple = random.choice(self.parent2different_items[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif c in self.parent2different_style:
                target_tuple = random.choice(self.parent2different_style[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif c in self.parent2different_silhouettes:
                target_tuple = random.choice(self.parent2different_silhouettes[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif c in self.parent2different_structures:
                target_tuple = random.choice(self.parent2different_structures[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
            elif c in self.parent2different_colors:
                target_tuple = random.choice(self.parent2different_colors[c])
                source_caption_id = self.caption2id[c]
                source_tuple = ("", source_caption_id)
                return self.add_single_turn(idx, source_tuple, target_tuple, mod_type)
        return

    def add_single_turn(self, idx, source_tuple, target_tuple, mod_type):
        source_word, source_caption_id = source_tuple
        target_word, target_caption_id = target_tuple
        target_caption = self.get_caption(target_caption_id)
        target_idx = random.choice(self.caption2imgids[target_caption])
        if source_word == "":
            mod_str = "add " + target_word
        else:
            mod_str = "replace " + source_word + " to " + target_word
        return {
            "source_img_id": idx,
            "target_img_id": target_idx,
            "source_word": source_word,
            "target_word": target_word,
            "mod_str": mod_str,
            "mod_type": mod_type
        }
