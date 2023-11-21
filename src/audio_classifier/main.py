import argparse

import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd

from audio_classifier.data import params
from audio_classifier.utils.yamnet import YAMNet
from audio_classifier.utils.utils import read_class_names
from audio_classifier.utils.preprocessing import preprocess_input
# from audio_classifier.utils.plot import Plotter


#################### MODEL #####################
# absolute paths
YAMNET_CLASSES = read_class_names("./src/audio_classifier/data/audioset_class_map.csv")

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = params.SAMPLE_RATE
WIN_SIZE_SEC = 0.975
CHUNK = int(WIN_SIZE_SEC * RATE)
RECORD_SECONDS = 500
MIC = None
PLT_CLASSES = list(range(len(YAMNET_CLASSES)))

################### CONFIG ###################
parser = argparse.ArgumentParser()
parser.add_argument(
    '--classes', 
    nargs='+',
    type=int,
    default=[13,15,70],
    # default=[],
    choices=PLT_CLASSES,
    help='Choose your target classes for inference.',
    )

args = parser.parse_args()

def main():
    """starts the listener and classifier.
    """
    ################### SETTINGS ###################
    plt_classes_choice = args.classes
    print(f'{plt_classes_choice=}')
    class_labels = True
    print(sd.query_devices())

    model = YAMNet(
        weights="./src/audio_classifier/data/yamnet.h5"
        ).get_model()

    #################### STREAM ####################
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(
        format=FORMAT,
        input_device_index=MIC,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    print("recording...")

    try:
        if plt_classes_choice:
            plt_classes = plt_classes_choice
            # plt_classes_lab = {k: v for k,v in YAMNET_CLASSES.items() if k in plt_classes_choice}
            # n_classes = len(plt_classes_choice)

        else:
            plt_classes = PLT_CLASSES
            # plt_classes_lab = YAMNET_CLASSES if class_labels else None
            # n_classes = len(YAMNET_CLASSES)

        # monitor = Plotter(n_classes=n_classes, fig_size=(12, 6), msd_labels=plt_classes_lab)

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            # Waveform
            data = preprocess_input(
                np.fromstring(stream.read(CHUNK), dtype=np.float32), RATE
            )
            prediction = model.predict(np.expand_dims(data, 0))[0]

            result = {
                YAMNET_CLASSES.get(i): round(prediction[i] * 100, 10)
                for i in plt_classes
            }
            print(f'{result=}')
            # print(f'{np.expand_dims(prediction[plt_classes], -1)=}')
            # monitor(data.transpose(), np.expand_dims(prediction[plt_classes], -1))

        print("finished recording")

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

    except (KeyboardInterrupt, SystemExit):
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print('KeyboardInterrupt')
        print('Recording stopped...')

    finally:
        stream.stop_stream()
        stream.close()

if __name__ == "__main__":
    main()
