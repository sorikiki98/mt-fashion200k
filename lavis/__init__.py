import os
import sys

from omegaconf import OmegaConf

from lavis.common.registry import registry

from lavis.models import *
from lavis.processors import *
from lavis.tasks import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)

repo_root = os.path.join(os.path.dirname(root_dir), "lavis_repo")
registry.register_path("repo_root", repo_root)

cache_root = os.path.expanduser(default_cfg.env.cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
