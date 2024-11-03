from utils.training_utils import prepare_training, train_model

def main(
    training_config_path: str
) -> None:
    """
    Main function to execute the training process.

    Args:
        training_config_path (str): Path to the training TOML configuration file.
    """
    # Prepare the training components
    model, criterion, optimizer, scheduler, early_stopping, config = prepare_training(training_config_path)

    # Start the training loop
    train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        config=config
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <training_config_path>")
    else:
        main(sys.argv[1])
