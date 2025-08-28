"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis import registry
from lavis import ObjaverseCaptionBuilder
from lavis import ObjaverseQADataset

@registry.register_builder("objaverse_mm_qa")
class ObjaverseQABuilder(ObjaverseCaptionBuilder):
    train_dataset_cls = ObjaverseQADataset
    eval_dataset_cls = ObjaverseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/objaverse/defaults_mm_qa.yaml",
    }