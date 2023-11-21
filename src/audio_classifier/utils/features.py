import numpy as np
import librosa
from audio_classifier.data import params

_MEL_BREAK_FREQUENCY_HERTZ: float = 700.0
_MEL_HIGH_FREQUENCY_Q: float = 1127.0


def _np_hann_periodic_window(length: int) -> np.ndarray:
    """
    Generates a Hann window of a given length.

    Args:
    - length (int): Length of the Hann window.

    Returns:
    - np.array: Hann window array.
    """
    if length == 1:
        return np.ones(1)
    odd = length % 2
    if not odd:
        length += 1
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / (length - 1))
    if not odd:
        window = window[:-1]
    return window


def _np_frame(data: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
    """
    Frames the input data using a given window length and hop length.

    Args:
    - data (np.array): Input data.
    - window_length (int): Length of the window.
    - hop_length (int): Hop length.

    Returns:
    - np.array: Framed data.
    """
    num_frames = 1 + int(np.floor((len(data) - window_length) // hop_length))
    shape = (num_frames, window_length)
    strides = (data.strides[0] * hop_length, data.strides[0])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def _np_stft(data: np.ndarray, fft_length: int, hop_length: int, window_length: int) -> np.ndarray:
    """
    Computes the Short-Time Fourier Transform (STFT) of the input data.

    Args:
    - data (np.array): Input data.
    - fft_length (int): Length of the FFT.
    - hop_length (int): Hop length.
    - window_length (int): Length of the window.

    Returns:
    - np.array: STFT result.
    """
    frames = _np_frame(data, window_length, hop_length)
    window = _np_hann_periodic_window(window_length)
    return np.fft.rfft(frames * window, fft_length)


def spec(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Computes the spectrogram of the input waveform.

    Args:
    - waveform (np.array): Input audio waveform.
    - sr (int): Sample rate of the audio waveform.

    Returns:
    - np.array: Spectrogram of the waveform.
    """
    win_samples: int = int(round(params.SAMPLE_RATE * params.STFT_WINDOW_SECONDS))
    hop_samples: int = int(round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
    n_fft: int = 2 ** int(np.ceil(np.log(win_samples) / np.log(2.0)))

    inp = (
        waveform
        if sr == params.SAMPLE_RATE
        else librosa.resample(waveform, sr, params.SAMPLE_RATE)
    )

    return np.abs(_np_stft(waveform, n_fft, hop_samples, win_samples))


def hertz_to_mel(frequencies_hertz: float) -> float:
    """
    Converts frequencies from Hertz to Mel scale.

    Args:
    - frequencies_hertz (float or np.array): Frequencies in Hertz.

    Returns:
    - float or np.array: Frequencies in Mel scale.
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
    )


def spectrogram_to_mel_matrix(
    num_mel_bins: int = 20,
    num_spectrogram_bins: int = 129,
    audio_sample_rate: int = 8000,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3800.0,
    unused_dtype: any = None,
) -> np.ndarray:
    """
    Computes the Mel conversion matrix for spectrogram bins.

    Args:
    - num_mel_bins (int): Number of Mel bins.
    - num_spectrogram_bins (int): Number of spectrogram bins.
    - audio_sample_rate (int): Sample rate of the audio.
    - lower_edge_hertz (float): Lower edge of the Mel filter bank.
    - upper_edge_hertz (float): Upper edge of the Mel filter bank.
    - unused_dtype (None): Unused parameter.

    Returns:
    - np.array: Mel conversion matrix.
    """
    nyquist_hertz: float = audio_sample_rate / 2.0
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            f"lower_edge_hertz {lower_edge_hertz:.1f} >= upper_edge_hertz {upper_edge_hertz:.1f}"
        )
    spectrogram_bins_hertz: np.ndarray = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel: np.ndarray = hertz_to_mel(spectrogram_bins_hertz)

    band_edges_mel: np.ndarray = np.linspace(
        hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2
    )

    mel_weights_matrix: np.ndarray = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]

        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (
            center_mel - lower_edge_mel
        )
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (
            upper_edge_mel - center_mel
        )
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope, upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def mel(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Computes the Mel spectrogram of the input waveform.

    Args:
    - waveform (np.array): Input audio waveform.
    - sr (int): Sample rate of the audio waveform.

    Returns:
    - np.array: Mel spectrogram of the waveform.
    """
    spectro: np.ndarray = spec(waveform, sr)
    mel_basis: np.ndarray = spectrogram_to_mel_matrix(
        params.MEL_BANDS,
        spectro.shape[1],
        params.SAMPLE_RATE,
        params.MEL_MIN_HZ,
        params.MEL_MAX_HZ,
    )
    return np.log(np.dot(spectro, mel_basis) + params.LOG_OFFSET)
