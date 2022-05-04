import os
import torch
import torchaudio
import dataset
import model
from os import path as osp
from torchsummary import summary


if __name__ == "__main__":
    AUDIO_DIR = osp.abspath("./dataset")
    ANNOTS_FILE_PATH = osp.join(AUDIO_DIR, "train_list.txt")
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 22050
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f"device is {device}")
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    ds = dataset.AudioDataset(
        annotations_file_path=ANNOTS_FILE_PATH,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectogram,
        sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device
    )
    # print(len(ds))
    signal, sr = ds[100]
    # print(signal.size())
    # print(signal.get_device())
    cnn = model.VGG(device=device)
    summary(model=cnn, input_size=signal.size(), device=device)
