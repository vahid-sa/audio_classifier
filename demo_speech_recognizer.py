import torch
import torchaudio
from os import path as osp
from pathlib import Path


LABELS = ['1', '3', '4', '5', '7', '8', '9', '11', '13', '15', '17', '19']


class Recognizer:
    """Audio words classifier"""

    def __init__(self) -> None:
        """Instance constructor"""
        self.model = None
        self.sample_rate = 8000

    def load_model(self) -> None:
        """Loads the trained model.

        Raises:
            FileNotFoundError: if the model is not placed aside of the script.
        """
        model_path = osp.join(osp.dirname(Path(__file__).absolute()), "model.pt")
        if not osp.isfile(model_path):
            raise FileNotFoundError("The model file not found.")
        model = torch.load(model_path)
        model.eval()
        self.model = model

    def predict(self, path: str) -> str:
        """Predicts the input audion file path class.

        Args:
            path (str): input audio file path

        Raises:
            FileNotFoundError: If path is not a file

        Returns:
            str: Predicted class for the input audio
        """
        if not osp.isfile(path):
            raise FileNotFoundError("The input file not found.")
        assert self.model is not None, "Please call load_model function at the begining."
        signal, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        with torch.no_grad():
            prediction = self.model(torch.unsqueeze(signal, dim=0))
        index = torch.argmax(prediction)
        label = LABELS[index]
        return label
