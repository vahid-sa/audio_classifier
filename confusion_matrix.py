import torch
import torchaudio
from sklearn.metrics import confusion_matrix, accuracy_score
from os import path as osp
from torch.utils.data import DataLoader
import seaborn as sns
from matplotlib import pyplot as plt
from dataset import AudioDataset, collate_fn
from settings import LABELS


model = torch.load(osp.abspath("./models/final_model.pt"))
model.eval()
dataset = AudioDataset(
    annotations_file_path=osp.abspath("./dataset/annotation_list.txt"),
    audio_dir=osp.abspath("./dataset/"),
    sample_rate=8000,
    num_samples=22050,
    device="cpu",
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=len(dataset),
    collate_fn=collate_fn,
)
for data in dataloader:
    inputs, targets = data
    with torch.no_grad():
        predictions = model(inputs)
    predictions = torch.argmax(torch.squeeze(predictions), dim=1)
    targets = targets.tolist()
    predictions = predictions.tolist()
m = confusion_matrix(y_true=targets, y_pred=predictions)
ax = sns.heatmap(m, annot=True, cmap='Blues')
ax.xaxis.set_ticklabels(LABELS)
ax.yaxis.set_ticklabels(LABELS)
plt.title("train+validation")
print(accuracy_score(targets, predictions))
# plt.savefig("train.png")
plt.show()
