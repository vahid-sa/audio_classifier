import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import AudioDataset
from evaluate import evaluate

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
    validation_dataset: AudioDataset,
    num_epochs: int, batch_size: int,
    learning_rate: float,
    device: str,
    save_state_dict_path: str,
):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset), num_workers=2,
    )
    training_dataloader_for_eval = DataLoader(
        training_dataset, batch_size=len(training_dataset), num_workers=2,
    )
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(training_dataloader):
            inputs, targets = data
            predictions = model(inputs)
            loss = loss_fn(predictions, targets.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch} | iteration: {i} | loss: {loss.item()}")
        model.eval()
        training_accuracy = evaluate(model, training_dataloader_for_eval, device)
        validation_accruacy = evaluate(model, validation_dataloader, device)
        print(f"Epoch {epoch} accuracy:: training: {training_accuracy} | validation: {validation_accruacy}")
        print()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        save_state_dict_path,
    )
