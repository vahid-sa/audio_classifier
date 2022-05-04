import os
import torch
import torchaudio
import dataset
import model
from os import path as osp
from torch import nn
from torchsummary import summary
from train import train


if __name__ == "__main__":
    AUDIO_DIR = osp.abspath("./dataset")
    ANNOTS_FILE_PATH = osp.join(AUDIO_DIR, "train_list.txt")
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 22050
    BATCH_SIZE = 1
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    SAVE_STATE_DICT_PATH = osp.abspath("./model_state_dict")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"device is {device}")
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    training_dataset = dataset.AudioDataset(
        annotations_file_path=ANNOTS_FILE_PATH,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectogram,
        sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device,
    )
    # print(len(ds))
    signal, sr = training_dataset[100]
    # print(signal.size())
    # print(signal.get_device())
    cnn = model.VGG(device=device)
    summary(model=cnn, input_size=signal.size(), device=device)

    # training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    # print(f"loss_fn: {type(loss_fn)} | optimizer: {type(optimizer)}")
    train(
        model=cnn,
        training_dataset=training_dataset,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=device,
        save_state_dict_path=SAVE_STATE_DICT_PATH,
    )
