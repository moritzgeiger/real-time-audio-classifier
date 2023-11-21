FROM python:3.9

RUN apt-get update -y && apt-get install -y build-essential libpq-dev
RUN apt-get install -y \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    # alsa-base \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils

# RUN apt-get install ffmpeg libav-tools -y

RUN pip install --upgrade pip

WORKDIR /app

# COPY requirements.txt .
# RUN pip install -r requirements.txt

COPY . .
RUN pip install .

# CMD ["/bin/bash"]
CMD ["python3", "./sound_event_detection.py"]
