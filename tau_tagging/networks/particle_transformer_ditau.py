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

## loss function definition
class CrossEntropyLogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction','loss_lambda','loss_gamma','quantiles']
    
    def __init__(self, reduction: str = 'mean', loss_lambda: float = 1., loss_gamma: float = 1., quantiles: list = []) -> None:
        super(CrossEntropyLogCoshLoss, self).__init__(None, None, reduction)
        self.loss_lambda = loss_lambda;
        self.loss_gamma = loss_gamma;
        self.quantiles = quantiles;

    def forward(self, input_cat: Tensor, y_cat: Tensor, input_reg: Tensor, y_reg: Tensor) -> Tensor:

        ## classification term
        loss_cat  = torch.nn.functional.cross_entropy(input_cat,y_cat,reduction=self.reduction);

        ## regression terms
        loss_mean  = 0;
        loss_quant = 0;
        loss_reg   = 0;
        x_reg      = input_reg-y_reg;

        for idx,q in enumerate(self.quantiles):
            if idx>0 or len(self.quantiles)>1:
                x_reg_eval = x_reg[:,idx]
            else:
                x_reg_eval = x_reg
            if q <= 0:
                loss_mean += x_reg_eval+torch.nn.functional.softplus(-2.*x_reg_eval)-math.log(2);
            elif q > 0:
                loss_quant += q*x_reg_eval*torch.ge(x_reg_eval,0)
                loss_quant += (q-1)*x_reg_eval*torch.less(x_reg_eval,0);

        if self.reduction == 'mean':
            loss_quant = loss_quant.mean();
            loss_mean = loss_mean.mean();
        elif self.reduction == 'sum':
            loss_quant = loss_quant.sum();
            loss_mean = loss_mean.sum();

        loss_reg = self.loss_lambda*loss_mean+self.loss_gamma*loss_quant;

        return loss_cat+loss_reg, loss_cat, loss_reg;


## get loss function
def get_loss(data_config, **kwargs):

    quantiles = data_config.target_quantile;

    return CrossEntropyLogCoshLoss(
        reduction = kwargs.get('reduction','mean'),
        loss_lambda = kwargs.get('loss_lambda',1),
        loss_gamma = kwargs.get('loss_gamma',1),
        quantiles = quantiles
    );
