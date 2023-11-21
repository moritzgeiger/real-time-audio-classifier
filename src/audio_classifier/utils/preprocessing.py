import numpy as np
import librosa

from audio_classifier.utils.features import mel
from audio_classifier.data import params


def preprocess_input(waveform: np.ndarray, sr: int):
    inp = (
        waveform
        if sr == params.SAMPLE_RATE
        else librosa.resample(waveform, sr, params.SAMPLE_RATE)
    )

    return mel(waveform, params.SAMPLE_RATE)
