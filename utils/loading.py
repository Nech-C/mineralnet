from typing import Union, List, Dict

import torch.nn as nn
import toml


def load_model(
    toml_path: str
):
    """load pretrained model from toml config with possible modification"""
    config = toml.load(toml_path)
    model = None
    untrained_weights = []
    # load model from model repo
    if 'torch' in config['from']:
        import torch
        model = torch.hub.load(
            **config['backbone'])
    else:
        print('Not implemented yet')

    # modify model
    if 'module_swap' in config:
        for swap in config['module_swap']:
            new_module = load_module(swap['new_module'])
            setattr(model, swap['target'], new_module)

    return {'model': model, 'untrained_weights': untrained_weights}


def load_module(
    module_config: Union[Dict, List[Dict]]
):
    """load module defined in toml config"""
    if isinstance(module_config, dict):
        module_class = getattr(nn, module_config['torch_module'])
        module = module_class(**module_config['args'])
        return module

    return nn.ModuleList([load_module(module) for module in module_config])


def prepare_training(
    training_config_path: str
):
    config = toml.load(training_config_path)
    model = load_model(config['model']['config_path'])
    
    
    
    return model
