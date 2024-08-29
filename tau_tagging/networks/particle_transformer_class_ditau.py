import numpy as np
import math
import torch
from torch import Tensor
from nn.model.ParticleTransformer import ParticleTransformerTagger

## get model
def get_model(data_config, **kwargs):

    ## number of classes
    num_classes = len(data_config.label_value)
    ## number of targets                                                                                                                                                                     
    num_targets = 0;
    if type(data_config.target_value) == dict:
        num_targets = sum(len(dct) if type(dct) == list else 1 for dct in data_config.target_value.values())
    elif data_config.target_value == None: num_targets = 0;
    else:
        num_targets = len(data_config.target_value);

    ## options
    cfg = dict(
        pf_input_dim = len(data_config.input_dicts['pf_features']),
        sv_input_dim = len(data_config.input_dicts['sv_features']),
        lt_input_dim = len(data_config.input_dicts['lt_features']),
        num_classes = num_classes,
        num_targets = num_targets,
        num_domains = [],
        pair_input_dim = len(data_config.input_dicts['pf_vectors']),
        pair_extra_dim = 0,
        embed_dims = [128, 256, 128],
        pair_embed_dims = [64, 64, 64],
        block_params = None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        num_heads = kwargs.get('num_heads',8),
        num_layers = kwargs.get('num_layers',8),
        num_cls_layers = kwargs.get('num_cls_layers',2),
        remove_self_pair = kwargs.get('remove_self_pair',False),
        use_pre_activation_pair = kwargs.get('use_pre_activation_pair',True),
        activation = kwargs.get('activation','gelu'),
        trim = kwargs.get('use_trim',True),
        for_inference = kwargs.get('for_inference',False),
        alpha_grad = kwargs.get('alpha_grad',1),
        use_amp = kwargs.get('use_amp',False),
        split_domain_outputs = kwargs.get('split_domain_outputs',False),
        fc_params = [(256, 0.1), (128, 0.1), (96, 0.1), (64, 0.1)],
        fc_domain_params = []
    );

    ## model
    model = ParticleTransformerTagger(**cfg)

    ## model info
    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['output'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output':{0:'N'}}},
        }

    return model, model_info



## get loss function
def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
