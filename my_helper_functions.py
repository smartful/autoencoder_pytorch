import matplotlib.pyplot as plt

import torch
from torch import nn

from tqdm.auto import tqdm

def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss = 0
    
    # Loop through data loader data batches
    for batch, (X, _) in enumerate(dataloader):
        # Send data to target device
        X = X.to(device)

        # 1. Forward pass
        x_hat = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(x_hat, X)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss per batch 
    train_loss = train_loss / len(dataloader)
    return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss = 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, _) in enumerate(dataloader):
            # Send data to target device
            X = X.to(device)
    
            # 1. Forward pass
            x_hat = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(x_hat, X)
            test_loss += loss.item()

    # Adjust metrics to get average loss per batch 
    test_loss = test_loss / len(dataloader)
    return test_loss

# Il faudra enlever l'accuracy
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.MSELoss(),
          epochs: int = 5,
          device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "test_loss": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)

    # 6. Return the filled results at the end of the epochs
    return results


def plot_training_data(typology: str, train_data: list, test_data: list):
    plt.figure()
    plt.title(f"{typology}")
    plt.plot(range(len(train_data)), train_data, label=f"Training {typology}")
    plt.plot(range(len(test_data)), test_data, label=f"Test {typology}")
    plt.xlabel("epochs")
    plt.ylabel(f"{typology}")
    plt.legend()
    plt.show()