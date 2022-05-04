import os
import numpy as np
import torch
import torchaudio
from os import path as osp
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        annotations_file_path: str,
        audio_dir: str,
        transformation: object,
        sample_rate: int,
        num_samples: int,
        device: str,
    ) -> None:
        self.annotations_files = self._load_annotations_file(annotations_file_path)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = sample_rate
        self.num_samples = num_samples
        self.classes = {
            "1": 0, "3": 1, "4": 2, "5": 3, "7": 4, "8": 5, "9": 6, "11": 7, "13": 8, "15": 9, "17": 10, "19": 11,
        }

    def __len__(self) -> int:
        return len(self.annotations_files)

    def __getitem__(self, index: int) -> tuple:
        filename = self.annotations_files[index]
        path = osp.join(self.audio_dir, filename)
        label, classID = self._extract_class(filename)
        signal, sr = torchaudio.load(path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, classID

    def _cut_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        signal_length = signal.shape[1]
        if signal_length < self.num_samples:
            num_missing_samples = self.num_samples - signal_length
            num_paddings = (0, num_missing_samples)  # (left, right)
            signal = torch.nn.functional.pad(signal, num_paddings)
        return signal

    def _resample_if_necessary(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    @staticmethod
    def _load_annotations_file(annotations_file_path: str) -> list:
        f = open(annotations_file_path, "r")
        annotations_string = f.read()
        f.close()
        annotations_files = annotations_string.splitlines()
        return annotations_files

    def _extract_class(self, filename: str) -> int:
        name: str = osp.basename(filename)
        label = name.split("(")[1].split(")")[0]
        id = self.classes[label]
        return label, id
