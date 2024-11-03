import toml
import wandb
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Dict, Iterator, Tuple
import os

def prepare_training(
    training_config_path: str
) -> Tuple[nn.Module, nn.Module, Optimizer, _LRScheduler, Any, Dict[str, Any]]:
    """
    Prepare the model, loss function, optimizer, scheduler, early stopping, and config.

    Args:
        training_config_path (str): Path to the training TOML configuration file.

    Returns:
        Tuple containing:
            - model (nn.Module): The model to train.
            - criterion (nn.Module): The loss function.
            - optimizer (Optimizer): The optimizer.
            - scheduler (_LRScheduler): The learning rate scheduler.
            - early_stopping (Any): The early stopping handler.
            - config (Dict[str, Any]): The training configuration.
    """
    # Load the training configuration
    config = toml.load(training_config_path)

    # Initialize wandb for logging
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb'].get('entity'),
        config=config  # Log the configuration
    )

    # Load the model using the model configuration
    from model_utils import load_model
    model = load_model(config['model']['config_path'])

    # Create the loss function
    criterion = create_loss_function(config['loss'])

    # Create the optimizer with parameters that require gradients
    optimizer = create_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        config['optimizer']
    )

    # Create the learning rate scheduler
    scheduler = create_scheduler(optimizer, config['scheduler'])

    # Initialize early stopping
    early_stopping = EarlyStopping(**config['early_stopping'])

    return model, criterion, optimizer, scheduler, early_stopping, config

def create_optimizer(
    model_parameters: Iterator[nn.Parameter],
    optimizer_config: Dict[str, Any]
) -> Optimizer:
    """
    Create an optimizer based on the configuration.

    Args:
        model_parameters (Iterator[nn.Parameter]): Parameters of the model to optimize.
        optimizer_config (Dict[str, Any]): Configuration dictionary for the optimizer.

    Returns:
        Optimizer: The instantiated optimizer.
    """
    # Import the torch.optim module
    import torch.optim as optim

    # Get the optimizer class from torch.optim
    optimizer_class = getattr(optim, optimizer_config['type'])
    # Get the parameters for the optimizer constructor
    optimizer_params = optimizer_config.get('params', {})
    # Instantiate the optimizer with the model parameters and provided arguments
    optimizer = optimizer_class(model_parameters, **optimizer_params)
    return optimizer

def create_scheduler(
    optimizer: Optimizer,
    scheduler_config: Dict[str, Any]
) -> _LRScheduler:
    """
    Create a learning rate scheduler based on the configuration.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        scheduler_config (Dict[str, Any]): Configuration dictionary for the scheduler.

    Returns:
        _LRScheduler: The instantiated learning rate scheduler.
    """
    # Import the torch.optim.lr_scheduler module
    import torch.optim.lr_scheduler as lr_scheduler

    # Get the scheduler class from torch.optim.lr_scheduler
    scheduler_class = getattr(lr_scheduler, scheduler_config['type'])
    # Get the parameters for the scheduler constructor
    scheduler_params = scheduler_config.get('params', {})
    # Instantiate the scheduler with the optimizer and provided arguments
    scheduler = scheduler_class(optimizer, **scheduler_params)
    return scheduler

def create_loss_function(
    loss_config: Dict[str, Any]
) -> nn.Module:
    """
    Create a loss function based on the configuration.

    Args:
        loss_config (Dict[str, Any]): Configuration dictionary for the loss function.

    Returns:
        nn.Module: The instantiated loss function.
    """
    # Get the loss function class from torch.nn
    loss_class = getattr(nn, loss_config['type'])
    # Get the parameters for the loss function constructor
    loss_params = loss_config.get('params', {})
    # Instantiate the loss function with the provided arguments
    criterion = loss_class(**loss_params)
    return criterion

class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.

    Attributes:
        patience (int): How long to wait after last time validation loss improved.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        best_loss (float): Best recorded validation loss.
        counter (int): Number of epochs since last improvement.
        early_stop (bool): Whether training should be stopped.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0
    ):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait before stopping.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(
        self,
        val_loss: float
    ) -> bool:
        """
        Check if validation loss has improved; update counter and early_stop flag.

        Args:
            val_loss (float): Current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        # If best_loss is None, this is the first validation; set best_loss
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        # Check if validation loss has improved by more than delta
        if val_loss < self.best_loss - self.delta:
            # Improvement found; reset counter and update best_loss
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # No improvement; increment counter
            self.counter += 1
            if self.counter >= self.patience:
                # Patience exceeded; set early_stop flag
                self.early_stop = True
                return True
            return False

def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    early_stopping: EarlyStopping,
    config: Dict[str, Any]
) -> None:
    """
    Train the model using the provided components and configuration.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        scheduler (_LRScheduler): The learning rate scheduler.
        early_stopping (EarlyStopping): Early stopping handler.
        config (Dict[str, Any]): Training configuration parameters.
    """
    from torch.utils.data import DataLoader

    # Placeholder for dataset class; replace with actual implementation
    class YourDataset(torch.utils.data.Dataset):
        def __init__(self, data_path):
            # Initialize dataset (e.g., load data, preprocess)
            pass

        def __len__(self):
            # Return the total number of samples
            return 0

        def __getitem__(self, idx):
            # Return a single sample (input and target)
            return None, None

    # Set up data loaders for training and validation data
    train_loader = DataLoader(
        dataset=YourDataset(config['data']['train_data']),
        batch_size=config['data']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=YourDataset(config['data']['val_data']),
        batch_size=config['data']['batch_size'],
        shuffle=False
    )

    num_epochs = config.get('num_epochs', 100)

    # Begin training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Initialize training metrics
        train_loss = 0.0

        for inputs, targets in train_loader:
            # Move data to appropriate device (e.g., GPU) if needed
            # inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss += loss.item() * inputs.size(0)

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to appropriate device if needed
                # inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)

        # Step the scheduler
        scheduler.step()

        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Check for early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered.")
            break

        # Save checkpoint if needed
        if (epoch + 1) % config['checkpoint']['save_freq'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # Include scheduler state if needed
            }, config['checkpoint']['dir'], epoch + 1)

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    epoch: int
) -> None:
    """
    Save the training checkpoint.

    Args:
        state (Dict[str, Any]): State dictionary containing model and optimizer states.
        checkpoint_dir (str): Directory where the checkpoint will be saved.
        epoch (int): Current epoch number.
    """
    # Create the checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Define the checkpoint file path
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    # Save the checkpoint
    torch.save(state, checkpoint_path)
    # Optionally, save the model to wandb
    wandb.save(checkpoint_path)
