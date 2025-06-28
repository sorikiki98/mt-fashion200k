import numpy as np
from PIL import Image
import re
import torch
import torch.utils.data
import random
from fashion_words import single_words, combinations, colors, items, material, pattern, functionalities, silhouettes, \
    structures, details, season, style, general
from dataset_utils import extract_siblings_per_category


class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset."""

    def __init__(self):
        super(BaseDataset, self).__init__()
        self.imgs = []
        self.train_queries = []

    def get_loader(self,
                   batch_size,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_train_queries(self):
        return self.train_queries

    def get_all_texts(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError

    def get_img(self, idx, raw_img=False):
        raise NotImplementedError


class Fashion200k(BaseDataset):
    """Fashion200k dataset."""

    def __init__(self, path, split="test", transform=None):
        super(Fashion200k, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + "/"

        # get label files for the split
        label_path = path + "/labels/"
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []

        def caption_post_process(s):
            return s.strip().replace(".",
                                     "dotmark").replace("?", "questionmark").replace(
                "&", "andmark").replace("*", "starmark")

        for filename in label_files:
            print("read " + filename)
            with open(label_path + "/" + filename, "rt", encoding="UTF8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.split("	")
                img = {
                    "file_path": line[0],
                    "detection_score": line[1],
                    "captions": [caption_post_process(line[2])],
                    "split": split,
                    "modifiable": False
                }
                self.imgs += [img]
        print("Fashion200k:", len(self.imgs), "images")

        self.caption_index_init_()

    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = "replace " + source_word + " with " + target_word
        return source_word, target_word, mod_str

    def generate_train_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img["file_path"]] = i
        with open(self.img_path + "/train_queries.txt") as f:
            lines = f.readlines()
        self.train_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[idx]["captions"][0]
            target_caption = self.imgs[target_idx]["captions"][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.train_queries += [{
                "source_img_id": idx,
                "source_caption": source_caption,
                "target_caption": target_caption,
                "mod": {
                    "str": mod_str
                }
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        attribute2category = {}

        for i, img in enumerate(self.imgs):
            for c in img["captions"]:
                for key, value in combinations.items():
                    if " " + key + " " in c:
                        c = c.replace(" " + key + " ", " " + value + " ")
                    if len(c.split()) > 1 and c.split()[-1] == key:
                        c = " ".join(c.split()[:-1]) + " " + value
                    if len(c.split()) > 1 and c.split()[0] == key:
                        c = value + " " + " ".join(c.split()[1:])
                for key, value in single_words.items():
                    if " " + key + " " in c:
                        c = c.replace(" " + key + " ", " " + value + " ")
                    if len(c.split()) > 1 and c.split()[-1] == key:
                        c = " ".join(c.split()[:-1]) + " " + value
                    if len(c.split()) > 1 and c.split()[0] == key:
                        c = value + " " + " ".join(c.split()[1:])
                for w in c.strip().split():
                    if not (w in colors or w in items or w in material or w in pattern or w in functionalities
                            or w in silhouettes or w in structures or w in details or w in season or w in style
                            or w in general):
                        c = re.sub(r'(^|\s)' + re.escape(w) + r'(\s|$)', ' ', c)
                        c = c.replace("  ", " ").strip()
                    if w not in attribute2category:
                        if w in colors:
                            attribute2category[w] = "color"
                        elif w in items:
                            attribute2category[w] = "item"
                        elif w in material:
                            attribute2category[w] = "material"
                        elif w in pattern:
                            attribute2category[w] = "pattern"
                        elif w in functionalities:
                            attribute2category[w] = "functionality"
                        elif w in silhouettes:
                            attribute2category[w] = "silhouette"
                        elif w in structures:
                            attribute2category[w] = "structure"
                        elif w in details:
                            attribute2category[w] = "detail"
                        elif w in season:
                            attribute2category[w] = "season"
                        elif w in style:
                            attribute2category[w] = "style"
                        elif w in general:
                            attribute2category[w] = "general"
                img["captions"] = [c]
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        self.attribute2category = attribute2category
        self.id2caption = id2caption
        self.caption2id = caption2id

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        parent2different_words = {}
        for c, i in caption2id.items():
            for w in c.split():
                p = c.replace(w, "")
                p = p.replace("  ", " ").strip()
                category = self.attribute2category[w]
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
                if p not in parent2different_words:
                    parent2different_words[p] = dict()
                if category not in parent2different_words[p]:
                    parent2different_words[p][category] = []
                if w not in [t[0] for t in parent2different_words[p][category]]:
                    parent2different_words[p][category].append((w, i))

        self.parent2children_captions = parent2children_captions
        self.parent2different_words = parent2different_words
        self.parent2different_colors = extract_siblings_per_category(parent2different_words, "color")
        self.parent2different_items = extract_siblings_per_category(parent2different_words, "item")
        self.parent2different_material = extract_siblings_per_category(parent2different_words, "material")
        self.parent2different_pattern = extract_siblings_per_category(parent2different_words, "pattern")
        self.parent2different_functionalities = extract_siblings_per_category(parent2different_words, "functionality")
        self.parent2different_silhouettes = extract_siblings_per_category(parent2different_words, "silhouette")
        self.parent2different_structures = extract_siblings_per_category(parent2different_words, "structure")
        self.parent2different_details = extract_siblings_per_category(parent2different_words, "detail")
        self.parent2different_season = extract_siblings_per_category(parent2different_words, "season")
        self.parent2different_style = extract_siblings_per_category(parent2different_words, "style")
        self.parent2different_general = extract_siblings_per_category(parent2different_words, "general")

        # identify parent captions for each image
        for img in self.imgs:
            img["modifiable"] = False
            img["parent_captions"] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]["modifiable"] = True
                        self.imgs[imgid]["parent_captions"] += [p]
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img["modifiable"]:
                num_modifiable_imgs += 1
        print("Modifiable images", num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]["modifiable"]:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img["parent_captions"])
            c = random.choice(self.parent2children_captions[p])
            if c not in img["captions"]:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]["captions"][0]
        target_caption = self.imgs[target_idx]["captions"][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img["captions"]:
                texts.append(c)
        return texts

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
            idx)
        out = {}
        out["source_img_id"] = idx
        out["source_img_data"] = self.get_img(idx)
        out["source_caption"] = self.imgs[idx]["captions"][0]
        out["target_img_id"] = target_idx
        out["target_img_data"] = self.get_img(target_idx)
        out["target_caption"] = self.imgs[target_idx]["captions"][0]
        out["mod"] = {"str": mod_str}
        return out

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.imgs[idx]["file_path"]
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img.show()
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img

    def get_caption(self, caption_idx):
        caption = self.id2caption[caption_idx]
        return caption
