import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import AudioDataset

def train_one_epoch(model, dataloader: DataLoader, loss_fn, optimizer, epoch: int):
    for i, data in enumerate(dataloader):
        inputs, targets = data
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch} | iteration: {i} | loss: {loss.item()}")


def train(
    model: nn.Module,
    training_dataset: AudioDataset,
    num_epochs: int, batch_size: int,
    learning_rate: float,
    device: str,
    save_state_dict_path: str,
):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True,
    )
    for epoch in range(num_epochs):
        for i, data in enumerate(training_dataloader):
            inputs, targets = data
            predictions = model(inputs)
            loss = loss_fn(predictions, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch} | iteration: {i} | loss: {loss.item()}")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        save_state_dict_path,
    )
