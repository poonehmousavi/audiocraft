# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset of music tracks with rich metadata.
"""
from dataclasses import dataclass, field, fields, replace
import gzip
import json
import logging
from pathlib import Path
import random
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
from ..utils.utils import warn_once


logger = logging.getLogger(__name__)

{"speaker_emotion": "angry", "speech_file": "/app/speech/ttas/new_dataset/speech/c5nShR4Lucw.wav"}

@dataclass
class SpeechSoundInfo(AudioInfo):
    """Segment info augmented with music metadata.
    """
    # music-specific metadata
    speech_text: tp.Optional[str] = None
    audio_cap: tp.Optional[str] = None  
    reasoning: tp.Optional[str] = None
    speaker_gender: tp.Optional[str] = None
    speaker_emotion: tp.Optional[str] = None
    # Eaither  speech or Audio based on what is the input wav
    self_wav: tp.Optional[WavCondition] = None


    @property
    def has_music_meta(self) -> bool:
        return self.name is not None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()
        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == 'self_wav':
                out.wav[key] = value
            else:
                if isinstance(value, list):
                    value = ' '.join(value)
                out.text[key] = value
        return out

    @staticmethod
    def attribute_getter(attribute):
        if attribute in ['speech_text', 'audio_cap', 'reasoning']:
            preprocess_func = get_string
        elif attribute in ['speaker_gender', 'speaker_emotion']:
            preprocess_func = get_keyword
        else:
            preprocess_func = None
        return preprocess_func

    @classmethod
    def from_dict(cls, dictionary: dict, fields_required: bool = False):
        _dictionary: tp.Dict[str, tp.Any] = {}

        # allow a subset of attributes to not be loaded from the dictionary
        # these attributes may be populated later
        post_init_attributes = ['self_wav']
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



class SpeechSoundDataset(InfoAudioDataset):
    """SpeechSound dataset is an AudioDataset with speech_sound metadata.

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
        breakpoint()

        self.info_fields_required = info_fields_required
        self.merge_text_p = merge_text_p
        self.drop_desc_p = drop_desc_p
        self.drop_other_p = drop_other_p
        self.joint_embed_attributes = joint_embed_attributes
        self.paraphraser = None
        if paraphrase_source is not None:
            self.paraphraser = Paraphraser(paraphrase_source, paraphrase_p)

    def __getitem__(self, index):
        wav, info = super().__getitem__(index)
        info_data = info.to_dict()
        music_info_path = Path(info.meta.path).with_suffix('.json')

        if Path(music_info_path).exists():
            # print(music_info_path)
            with open(music_info_path, 'r') as json_file:
                music_data = json.load(json_file)
                music_data.update(info_data)
                music_info = MusicInfo.from_dict(music_data, fields_required=self.info_fields_required)
            if self.paraphraser is not None:
                music_info.description = self.paraphraser.sample(music_info.meta.path, music_info.description)
            if self.merge_text_p:
                music_info = augment_music_info_description(
                    music_info, self.merge_text_p, self.drop_desc_p, self.drop_other_p)
        else:
            music_info = MusicInfo.from_dict(info_data, fields_required=False)
        # print(music_info)

        music_info.self_wav = WavCondition(
            wav=wav[None], length=torch.tensor([info.n_frames]),
            sample_rate=[info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])

        for att in self.joint_embed_attributes:
            att_value = getattr(music_info, att)
            joint_embed_cond = JointEmbedCondition(
                wav[None], [att_value], torch.tensor([info.n_frames]),
                sample_rate=[info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])
            music_info.joint_embed[att] = joint_embed_cond

        return wav, music_info


def get_musical_key(value: tp.Optional[str]) -> tp.Optional[str]:
    """Preprocess key keywords, discarding them if there are multiple key defined."""
    if value is None or (not isinstance(value, str)) or len(value) == 0 or value == 'None':
        return None
    elif ',' in value:
        # For now, we discard when multiple keys are defined separated with comas
        return None
    else:
        return value.strip().lower()


def get_bpm(value: tp.Optional[str]) -> tp.Optional[float]:
    """Preprocess to a float."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
