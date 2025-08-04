# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset of music tracks with rich metadata.
"""
from dataclasses import dataclass, field, fields, replace
import logging
from pathlib import Path
import typing as tp

import torch

from .info_audio_dataset import (
    InfoAudioDataset,
    AudioInfo,
    get_keyword_list,
    get_keyword,
    get_string
)
from ..modules.conditioners import (
    ConditioningAttributes,
    JointEmbedCondition,
    WavCondition,
)



logger = logging.getLogger(__name__)


@dataclass
class SoundInfo(AudioInfo):
    """Segment info augmented with music metadata.
    """
    # music-specific metadata
    speech_text: tp.Optional[str] = None
    audio_cap: tp.Optional[str] = None  # anonymized artist id, used to ensure no overlap between splits
    speaker_emotion: tp.Optional[str] = None
    speaker_gender: tp.Optional[str] = None
    reasoning: tp.Optional[str] = None
    self_wav: tp.Optional[WavCondition] = None

    @property
    def has_sound_meta(self) -> bool:
        return self.name is not None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()
        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == 'self_wav':
                out.wav[key] = value
            elif key == 'joint_embed':
                for embed_attribute, embed_cond in value.items():
                    out.joint_embed[embed_attribute] = embed_cond
            else:
                if isinstance(value, list):
                    value = ' '.join(value)
                out.text[key] = value
        return out

    @staticmethod
    def attribute_getter(attribute):
        if attribute in ['speaker_emotion', 'speaker_gender']:
            preprocess_func = get_keyword
        elif attribute in ['reasoning','speech_text', 'audio_cap']:
            preprocess_func = get_string
        else:
            preprocess_func = None
        return preprocess_func

    @classmethod
    def from_dict(cls, dictionary: dict, fields_required: bool = False):
        _dictionary: tp.Dict[str, tp.Any] = {}

        # allow a subset of attributes to not be loaded from the dictionary
        # these attributes may be populated later
        post_init_attributes = ['self_wav', 'joint_embed']
        optional_fields = ['keywords']

        for _field in fields(cls):
            if _field.name in post_init_attributes:
                continue
            elif _field.name not in dictionary:
                if fields_required and _field.name not in optional_fields:
                    raise KeyError(f"Unexpected missing key: {_field.name}")
            else:
                preprocess_func: tp.Optional[tp.Callable] = cls.attribute_getter(_field.name)
                value = dictionary[_field.name]
                if preprocess_func:
                    value = preprocess_func(value)
                _dictionary[_field.name] = value
        return cls(**_dictionary)


class SoundDataset(InfoAudioDataset):
    """Music dataset is an AudioDataset with music-related metadata.

    Args:
        info_fields_required (bool): Whether to enforce having required fields.
        merge_text_p (float): Probability of merging additional metadata to the description.
        drop_desc_p (float): Probability of dropping the original description on text merge.
        drop_other_p (float): Probability of dropping the other fields used for text augmentation.
        joint_embed_attributes (list[str]): A list of attributes for which joint embedding metadata is returned.
        paraphrase_source (str, optional): Path to the .json or .json.gz file containing the
            paraphrases for the description. The json should be a dict with keys are the
            original info path (e.g. track_path.json) and each value is a list of possible
            paraphrased.
        paraphrase_p (float): probability of taking a paraphrase.

    See `audiocraft.data.info_audio_dataset.InfoAudioDataset` for full initialization arguments.
    """
    def __init__(self, *args, info_fields_required: bool = True,
                 merge_text_p: float = 0., drop_desc_p: float = 0., drop_other_p: float = 0.,
                 joint_embed_attributes: tp.List[str] = [],
                 paraphrase_source: tp.Optional[str] = None, paraphrase_p: float = 0,
                 **kwargs):
        kwargs['return_info'] = True  # We require the info for each song of the dataset.
        super().__init__(*args, **kwargs)
        self.info_fields_required = info_fields_required
        self.merge_text_p = merge_text_p
        self.drop_desc_p = drop_desc_p
        self.drop_other_p = drop_other_p
        self.joint_embed_attributes = joint_embed_attributes
        self.paraphraser = None


    def __getitem__(self, index):
        wav, info = super().__getitem__(index)
        info_data = info.to_dict()
        sound_info = SoundInfo.from_dict(info_data, fields_required=False)

        sound_info.self_wav = WavCondition(
            wav=wav[None], length=torch.tensor([info.n_frames]),
            sample_rate=[info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])
        
        return wav, sound_info



