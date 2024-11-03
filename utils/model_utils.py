import toml
import torch.nn as nn
from typing import Any, Dict

def load_model(
    toml_path: str
) -> nn.Module:
    """
    Load a pretrained model from a TOML config file with possible modifications.

    Args:
        toml_path (str): Path to the model TOML configuration file.

    Returns:
        nn.Module: The loaded and possibly modified model.
    """
    # Load the configuration from the TOML file
    config = toml.load(toml_path)
    model = None

    # Determine the source to load the model from
    if config['from'] == 'pytorch':
        import torch
        model = torch.hub.load(
            **config['backbone']  # Unpack the backbone configuration
        )
    else:
        # Handle other sources (e.g., 'huggingface', 'local')
        pass  # Not implemented yet

    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specified modules
    if 'unfreeze_modules' in config:
        for module_name in config['unfreeze_modules']:
            # Retrieve the module by name
            module = dict(model.named_modules()).get(module_name)
            if module:
                # Set requires_grad = True for all parameters in the module
                for param in module.parameters():
                    param.requires_grad = True
            else:
                # Handle the case where the module is not found
                print(f"Module '{module_name}' not found in the model.")

    # Swap modules if specified
    if 'module_swap' in config:
        for swap in config['module_swap']:
            # Load the new module based on the configuration
            new_module = load_module(swap['new_module'])
            # Replace the target module in the model
            setattr(model, swap['target'], new_module)

    return model

def load_module(
    module_config: Dict[str, Any]
) -> nn.Module:
    """
    Load a module defined in the TOML configuration.

    Args:
        module_config (Dict[str, Any]): Configuration dictionary for the module.

    Returns:
        nn.Module: The instantiated PyTorch module.
    """
    # Get the module class from torch.nn
    module_class = getattr(nn, module_config['torch_module'])
    # Get the arguments for the module constructor
    module_args = module_config.get('args', {})
    # Instantiate the module with the provided arguments
    module = module_class(**module_args)
    return module
