FROM python:3.9

RUN apt-get update -y && apt-get install -y \
    build-essential \
    libpq-dev

RUN apt-get install -y \
    portaudio19-dev \
    vorbis-tools \
    sox \
    alsa-utils \
    libasound2 \
    libasound2-plugins \
    pulseaudio \
    pulseaudio-utils \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get install -y \
#     libasound-dev \
#     libportaudio2 \
#     libportaudiocpp0 \
#     portaudio19-dev


RUN pip install --upgrade pip
WORKDIR /app

# COPY requirements.txt .
# RUN pip install -r requirements.txt

COPY . .
RUN pip install .

CMD ["/bin/bash"]
# ENTRYPOINT [ "audio-classifier"]