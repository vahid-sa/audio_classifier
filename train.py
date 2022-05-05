import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import AudioDataset, collate_fn
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
    weight_decay: float,
    device: str,
    save_state_dict_path: str,
):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    training_dataloader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn,
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset), num_workers=2, collate_fn=collate_fn,
    )
    training_dataloader_for_eval = DataLoader(
        training_dataset, batch_size=len(training_dataset), num_workers=2, collate_fn=collate_fn,
    )
    for i in range(num_epochs):
        epoch = i + 1
        model.train()
        for j, data in enumerate(training_dataloader):
            iteration = j + 1
            inputs, targets = data
            predictions: torch.Tensor = model(inputs)
            # loss = loss_fn(predictions, targets.to(device))
            loss = F.nll_loss(predictions.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch} | iteration: {iteration} | loss: {loss.item()}")
        scheduler.step()
        model.eval()
        training_accuracy = evaluate(model, training_dataloader_for_eval, device)
        validation_accruacy = evaluate(model, validation_dataloader, device)
        print(f"Epoch {epoch} accuracy:: training: {training_accuracy} | validation: {validation_accruacy}")
        print()
        if epoch % 10 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                save_state_dict_path.format(f"epoch_{epoch}".zfill(3)),
            )
    torch.save(model, save_state_dict_path.format("final_model"))
