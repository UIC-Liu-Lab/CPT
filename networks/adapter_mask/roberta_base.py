import os
import pdb
from pathlib import Path

import torch
import torch.nn as nn
import sys
sys.path.append('./networks/adapter_mask')
from roberta_model import RobertaForSequenceClassification, RobertaForMaskedLM
from roberta_adapter import add_roberta_adapters
from common import AdapterMaskConfig

class RobertaMaskBasedModel:

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        for_end_task=False,
        use_prompt=True,
        **kwargs
    ):
        # Drop most of the args for now
        outputs = super().forward(
            attention_mask=attention_mask,
            input_ids=input_ids,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )
        return outputs


class RobertaMaskForMaskedLM(RobertaMaskBasedModel, RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)


class RobertaMaskForSequenceClassification(RobertaMaskBasedModel, RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)