"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis import load_dataset_config
from lavis import (
    COCOCapBuilder,
    MSRVTTCapBuilder,
    MSVDCapBuilder,
    VATEXCapBuilder,
    MSRVTTCapInstructBuilder,
    MSVDCapInstructBuilder,
    VATEXCapInstructBuilder,
    WebVid2MCapBuilder,
    WebVid2MCapInstructBuilder,
    VALORCaptionBuilder,
    VALORCaptionInstructBuilder,
    ViolinCapBuilder,
    ViolinCapInstructBuilder,
    VlepCaptionInstructBuilder, 
    VlepCaptionBuilder,
    YouCookCaptionBuilder,
    YouCookCaptionInstructBuilder,
    COINCaptionBuilder,
    COINCaptionInstructBuilder,
    CharadeCaptionBuilder,
    CharadeCaptionInstructBuilder,
    TextCapsCapBuilder,
    TextCapsCapInstructBuilder,
    Flickr30kCapBuilder,
    Flickr30kCapInstructBuilder

)
from lavis import (
    ConceptualCaption12MBuilder,
    ConceptualCaption12MInstructBuilder,
    ConceptualCaption3MBuilder,
    ConceptualCaption3MInstructBuilder,
    VGCaptionBuilder,
    VGCaptionInstructBuilder,
    SBUCaptionBuilder,
    SBUCaptionInstructBuilder,
    Laion400MBuilder,
    Laion400MInstructBuilder
)
from lavis import (
    NLVRBuilder,
    SNLIVisualEntailmentBuilder,
    SNLIVisualEntailmentInstructBuilder,
    ViolinEntailmentInstructBuilder,
    ViolinEntailmentBuilder,
    ESC50ClassificationBuilder
)
from lavis import ImageNetBuilder
from lavis import (
    MSRVTTQABuilder, 
    MSVDQABuilder,
    MSRVTTQAInstructBuilder,
    MSVDQAInstructBuilder,
    MusicAVQABuilder,
    MusicAVQAInstructBuilder
)

from lavis import (
    COCOVQABuilder,
    COCOVQAInstructBuilder,
    OKVQABuilder,
    OKVQAInstructBuilder,
    AOKVQABuilder,
    AOKVQAInstructBuilder,
    VGVQABuilder,
    VGVQAInstructBuilder,
    GQABuilder,
    GQAInstructBuilder,
    IconQABuilder,
    IconQAInstructBuilder,
    ScienceQABuilder,
    ScienceQAInstructBuilder,
    OCRVQABuilder,
    OCRVQAInstructBuilder,
    VizWizVQABuilder
)
from lavis import (
    MSRVTTRetrievalBuilder,
    DiDeMoRetrievalBuilder,
    COCORetrievalBuilder,
    Flickr30kBuilder,
)

from lavis import (
    AudioSetBuilder,
    AudioCapsCapBuilder,
    AudioSetInstructBuilder,
    AudioCapsInstructCapBuilder,
    WavCapsCapInstructBuilder,
    WavCapsCapBuilder
)

from lavis import (
    ObjaverseCaptionInstructBuilder,
    ShapenetCaptionInstructBuilder,
    ObjaverseCaptionBuilder,
    ShapenetCaptionBuilder
)
from lavis import ObjaverseQABuilder
from lavis import ModelNetClassificationBuilder

from lavis import AudioCapsQABuilder, ClothoQABuilder

from lavis import (
    AVSDDialBuilder, 
    AVSDDialInstructBuilder,
    YT8MDialBuilder,
    LLaVA150kDialInstructBuilder,
    VisDialBuilder,
    VisDialInstructBuilder
)
from lavis import BlipDiffusionFinetuneBuilder

from lavis import DiscrnImagePcBuilder, DiscrnAudioVideoBuilder

from lavis import registry

__all__ = [
    "BlipDiffusionFinetuneBuilder",
    "COCOCapBuilder",
    "COCORetrievalBuilder",
    "COCOVQABuilder",
    "ConceptualCaption12MBuilder",
    "ConceptualCaption3MBuilder",
    "DiDeMoRetrievalBuilder",
    "Flickr30kBuilder",
    "GQABuilder",
    "ImageNetBuilder",
    "MSRVTTCapBuilder",
    "MSRVTTQABuilder",
    "MSRVTTRetrievalBuilder",
    "MSVDCapBuilder",
    "MSVDQABuilder",
    "NLVRBuilder",
    "OKVQABuilder",
    "AOKVQABuilder",
    "SBUCaptionBuilder",
    "SNLIVisualEntailmentBuilder",
    "VATEXCapBuilder",
    "VGCaptionBuilder",
    "VGVQABuilder",
    "AVSDDialBuilder",
    "Laion400MBuilder",

    "ViolinCapBuilder",
    "ViolinEntailmentBuilder",
    "VlepCaptionBuilder",
    "YouCookCaptionBuilder",
    "COINCaptionBuilder",
    "CharadeCaptionBuilder",
    "YT8MDialBuilder",
    "IconQABuilder",
    "ScienceQABuilder",
    "VisDialBuilder",
    "OCRVQABuilder",
    "VizWizVQABuilder",
    "TextCapsCapBuilder",
    "Flickr30kCapBuilder",
    "AudioSetBuilder",
    "AudioCapsCapBuilder",
    "WavCapsCapBuilder",
    "WebVid2MCapBuilder",
    "VALORCaptionBuilder",
    "ObjaverseCaptionBuilder",
    "ShapenetCaptionBuilder",
    "ObjaverseQABuilder",
    "MusicAVQABuilder",
    "ESC50ClassificationBuilder",

    ## Instruction Builders
    "AOKVQAInstructBuilder",
    "OKVQAInstructBuilder",
    "AudioSetInstructBuilder",
    "AudioCapsInstructCapBuilder",
    "AudioCapsQABuilder",
    "WavCapsCapInstructBuilder",
    "ObjaverseCaptionInstructBuilder",
    "ShapenetCaptionInstructBuilder",
    "ModelNetClassificationBuilder",
    "ObjaverseCaptionInstructBuilder",
    "MSRVTTCapInstructBuilder",
    "MSVDCapInstructBuilder",
    "VATEXCapInstructBuilder",
    "WebVid2MCapInstructBuilder",
    "MSRVTTQAInstructBuilder",
    "MSVDQAInstructBuilder",
    "VALORCaptionInstructBuilder",
    "AVSDDialInstructBuilder",
    "VisDialInstructBuilder",
    "MusicAVQAInstructBuilder",
    "ViolinCapInstructBuilder",
    "ViolinEntailmentInstructBuilder",
    "VlepCaptionInstructBuilder", 
    "YouCookCaptionInstructBuilder",
    "COINCaptionInstructBuilder",
    "CharadeCaptionInstructBuilder",
    "COCOVQAInstructBuilder",
    "VGVQAInstructBuilder",
    "GQAInstructBuilder",
    "IconQAInstructBuilder",
    "SNLIVisualEntailmentInstructBuilder",
    "Laion400MInstructBuilder",
    "LLaVA150kDialInstructBuilder",
    "ScienceQAInstructBuilder",
    "OCRVQAInstructBuilder",
    "TextCapsCapInstructBuilder",
    "Flickr30kCapInstructBuilder",
    "ConceptualCaption12MInstructBuilder",
    "ConceptualCaption3MInstructBuilder",
    "VGCaptionInstructBuilder",
    "SBUCaptionInstructBuilder",
    "ClothoQABuilder",

    # DisCRN
    "DiscrnImagePcBuilder",
    "DiscrnAudioVideoBuilder"

]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
            data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()
