import torch
from torch import nn
from torch.utils.data import DataLoader



def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    is_correct = []
    for data in dataloader:
        with torch.no_grad():
            inputs, targets = data
            predictions: torch.Tensor = model(inputs)
            eq: torch.Tensor = torch.eq(torch.argmax(predictions, dim=1), targets.to(device))
            is_correct.extend(eq.detach().cpu().tolist())
    accuracy = sum(is_correct) / len(is_correct)
    return accuracy
