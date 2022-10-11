"""Roberta model with CPT CL-plugins."""
import logging
import math
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfOutput
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention

from networks.adapter_mask.common import AdapterMaskConfig, freeze_all_parameters

logging.basicConfig(level=logging.INFO)


class RobertaAdapter(nn.Module):
    def __init__(self, config: AdapterMaskConfig):
        super().__init__()
        self.fc1 = torch.nn.Linear(config.hidden_size, config.adapter_size)
        self.fc2 = torch.nn.Linear(config.adapter_size, config.hidden_size)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        h = self.activation(self.fc1(x))
        h = self.activation(self.fc2(h))
        return x + h
        # return h


class RobertaAdapterMask(RobertaAdapter):
    def __init__(self, config: AdapterMaskConfig):
        super().__init__(config)
        self.efc1 = torch.nn.Embedding(config.ntasks, config.adapter_size)
        self.efc2 = torch.nn.Embedding(config.ntasks, config.hidden_size)
        self.gate = torch.nn.Sigmoid()
        self.config = config
        self.smax = config.smax

    def forward(self, x, t, s, smax=400, add_residual=True, residual=None):
        residual = x if residual is None else residual
        gfc1, gfc2 = self.mask(t=t, s=s)
        h = self.get_feature(gfc1, gfc2, x)
        if add_residual:
            output = residual + h
        else:
            output = h

        return output

    def get_feature(self, gfc1, gfc2, x):
        h = self.activation(self.fc1(x))
        h = h * gfc1.expand_as(h)

        h = self.activation(self.fc2(h))
        h = h * gfc2.expand_as(h)

        return h

    def mask(self, t: torch.LongTensor, s: int = None):

        efc1 = self.efc1(t)
        efc2 = self.efc2(t)

        gfc1 = self.gate(s * efc1)
        gfc2 = self.gate(s * efc2)

        if s == self.smax:  # you want to use it for evluation
            gfc1 = (gfc1 > 0.5).float()
            gfc2 = (gfc2 > 0.5).float()

        return [gfc1, gfc2]


class RobertaAdaptedSelfOutput(nn.Module):
    def __init__(self,
                 self_output: BertSelfOutput,
                 config: AdapterMaskConfig):
        super(RobertaAdaptedSelfOutput, self).__init__()
        self.self_output = self_output
        self.adapter_mask = RobertaAdapterMask(config)
        self.mode = config.mode

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, t, s, **kwargs):
        if self.mode == "sequential":
            hidden_states = self.self_output.dense(hidden_states)
            hidden_states = self.self_output.dropout(hidden_states)
            hidden_states = self.adapter_mask(hidden_states, t, s)
        elif self.mode == "parallel":
            adapter_change = self.adapter_mask(input_tensor, t, s)
            hidden_states = self.self_output.dense(hidden_states)
            hidden_states = self.self_output.dropout(hidden_states)
            hidden_states = hidden_states + adapter_change
        hidden_states = self.self_output.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAdaptedSelfAttention(nn.Module):
    """For parallel adapter."""

    def __init__(self,
                 self_attn: RobertaSelfAttention,
                 config: AdapterMaskConfig):
        super(RobertaAdaptedSelfAttention, self).__init__()
        if config.mode != "parallel":
            raise ValueError("This class is tailored for parallel adapter!")
        self.self_attn = self_attn
        self.adapter_mask = RobertaAdapterMask(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            t=None,
            s=None,
            **kwargs,
    ):
        mixed_query_layer = self.self_attn.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.self_attn.transpose_for_scores(self.self_attn.key(encoder_hidden_states))
            value_layer = self.self_attn.transpose_for_scores(self.self_attn.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.self_attn.transpose_for_scores(self.self_attn.key(hidden_states))
            value_layer = self.self_attn.transpose_for_scores(self.self_attn.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.self_attn.transpose_for_scores(self.self_attn.key(hidden_states))
            value_layer = self.self_attn.transpose_for_scores(self.self_attn.value(hidden_states))

        query_layer = self.self_attn.transpose_for_scores(mixed_query_layer)

        if self.self_attn.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        cross_attn_output = self.adapter_mask(hidden_states, t=t, s=s, add_residual=False)  # For parallel adapter.

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.self_attn.position_embedding_type == "relative_key" or self.self_attn.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.self_attn.distance_embedding(
                distance + self.self_attn.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.self_attn.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.self_attn.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.self_attn.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.self_attn.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.self_attn.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = context_layer + cross_attn_output  # For parallel adapter.

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.self_attn.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


def adapt_roberta_self_output(config: AdapterMaskConfig):
    return lambda self_output: RobertaAdaptedSelfOutput(self_output, config=config)


def adapt_roberta_self_attn(config: AdapterMaskConfig):
    return lambda self_attn: RobertaAdaptedSelfAttention(self_attn, config=config)


def add_roberta_adapters(roberta_model: BertModel, config: AdapterMaskConfig) -> BertModel:
    attn_config, ffn_config = deepcopy(config), deepcopy(config)
    attn_config.adapter_size = attn_config.attn_adapter_size
    ffn_config.adapter_size = ffn_config.ffn_adapter_size

    if config.mode == "sequential":
        for layer in roberta_model.encoder.layer:
            layer.attention.output = adapt_roberta_self_output(
                attn_config)(layer.attention.output)
            layer.output = adapt_roberta_self_output(ffn_config)(layer.output)
    elif config.mode == "parallel":
        for layer in roberta_model.encoder.layer:
            layer.attention.self = adapt_roberta_self_attn(attn_config)(layer.attention.self)
            layer.output = adapt_roberta_self_output(ffn_config)(layer.output)
    return roberta_model


def unfreeze_roberta_adapters(roberta_model: nn.Module) -> nn.Module:
    # Unfreeze trainable parts — layer norms and adapters
    for name, sub_module in roberta_model.named_modules():
        if isinstance(sub_module, (RobertaAdapter, nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return roberta_model


def load_roberta_adapter_model(
        roberta_model: nn.Module,
        checkpoint: str = None,
        mode: str = "sequential",
        attn_adapter_size: int = 200,
        ffn_adapter_size: int = 512,
        ntasks: int = 5):
    adapter_config = AdapterMaskConfig(
        hidden_size=768,
        adapter_size=-1,  # Deprecated.
        adapter_act='relu',
        adapter_initializer_range=1e-2,
        ntasks=ntasks,
        smax=400,
        mode=mode,
        attn_adapter_size=attn_adapter_size,
        ffn_adapter_size=ffn_adapter_size,
    )
    roberta_model.roberta = add_roberta_adapters(
        roberta_model.roberta, adapter_config)

    # freeze the bert model, unfreeze adapter
    roberta_model.roberta = freeze_all_parameters(roberta_model.roberta)
    roberta_model.roberta = unfreeze_roberta_adapters(roberta_model.roberta)

    if checkpoint is not None and checkpoint != 'None':
        print("loading checkpoint...")
        model_dict = roberta_model.state_dict()
        pretrained_dict = torch.load(checkpoint, map_location='cpu')
        new_dict = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict.keys()}
        model_dict.update(new_dict)
        print('Total : {} params are loaded.'.format(len(pretrained_dict)))
        roberta_model.load_state_dict(model_dict)
        print("loaded finished!")
    else:
        print('No checkpoint is included')
    return roberta_model


def save_roberta_adapter_model(roberta_model: nn.Module, save_path: str, accelerator=None):
    model_dict = {k: v for k, v in roberta_model.state_dict().items()
                  if 'adapter' in k}
    if accelerator is not None:
        accelerator.save(model_dict, save_path)
    else:
        torch.save(model_dict, save_path)


def forward(self, t, input_ids, segment_ids, input_mask, s=None):
    output_dict = {}

    sequence_output, pooled_output = \
        self.bert(input_ids=input_ids, token_type_ids=segment_ids,
                  attention_mask=input_mask, t=t, s=s)
    masks = self.mask(t, s)
    pooled_output = self.dropout(pooled_output)

    y = self.last(sequence_output)
    output_dict['y'] = y
    output_dict['masks'] = masks
    return output_dict


def forward_cls(self, t, input_ids, segment_ids, input_mask, start_mixup=None, s=None, l=None, idx=None, mix_type=None):
    output_dict = {}

    sequence_output, pooled_output = \
        self.bert(input_ids=input_ids, token_type_ids=segment_ids,
                  attention_mask=input_mask, t=t, s=s)
    masks = self.mask(t, s)
    pooled_output = self.dropout(pooled_output)

    y = self.last_cls(pooled_output)
    output_dict['y'] = y
    output_dict['masks'] = masks
    return output_dict


def mask(roberta_model, t, s, adapter_type="sequential"):
    masks = {}
    for layer_id in range(len(roberta_model.roberta.encoder.layer)):
        if adapter_type == "sequential":
            fc1_key = 'roberta.encoder.layer.' + \
                      str(layer_id) + '.attention.output.adapter_mask.fc1'  # gfc1
            fc2_key = 'roberta.encoder.layer.' + \
                      str(layer_id) + '.attention.output.adapter_mask.fc2'  # gfc2

            masks[fc1_key], masks[fc2_key] = roberta_model.roberta.encoder.layer[
                layer_id].attention.output.adapter_mask.mask(
                t, s)
        else:
            fc1_key = 'roberta.encoder.layer.' + \
                      str(layer_id) + '.attention.self.adapter_mask.fc1'  # gfc1
            fc2_key = 'roberta.encoder.layer.' + \
                      str(layer_id) + '.attention.self.adapter_mask.fc2'  # gfc2

            masks[fc1_key], masks[fc2_key] = roberta_model.roberta.encoder.layer[
                layer_id].attention.self.adapter_mask.mask(
                t, s)

        fc1_key = 'roberta.encoder.layer.' + \
                  str(layer_id) + '.output.adapter_mask.fc1'  # gfc1
        fc2_key = 'roberta.encoder.layer.' + \
                  str(layer_id) + '.output.adapter_mask.fc2'  # gfc2

        masks[fc1_key], masks[fc2_key] = roberta_model.roberta.encoder.layer[layer_id].output.adapter_mask.mask(
            t, s)

    return masks


def get_view_for(model, n, p, masks):
    for layer_id in range(12):
        if n == 'roberta.encoder.layer.' + str(layer_id) + '.attention.output.adapter_mask.fc1.weight':
            return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.attention.output.adapter_mask.fc1.bias':
            return masks[n.replace('.bias', '')].data.view(-1)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.attention.output.adapter_mask.fc2.weight':
            post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
            return torch.min(post, pre)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.attention.output.adapter_mask.fc2.bias':
            return masks[n.replace('.bias', '')].data.view(-1)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.output.adapter_mask.fc1.weight':
            # print('not nont')
            return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.output.adapter_mask.fc1.bias':
            return masks[n.replace('.bias', '')].data.view(-1)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.output.adapter_mask.fc2.weight':
            post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
            return torch.min(post, pre)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.output.adapter_mask.fc2.bias':
            return masks[n.replace('.bias', '')].data.view(-1)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.attention.self.adapter_mask.fc1.weight':
            return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.attention.self.adapter_mask.fc1.bias':
            return masks[n.replace('.bias', '')].data.view(-1)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.attention.self.adapter_mask.fc2.weight':
            post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
            return torch.min(post, pre)
        elif n == 'roberta.encoder.layer.' + str(layer_id) + '.attention.self.adapter_mask.fc2.bias':
            return masks[n.replace('.bias', '')].data.view(-1)
    return None
