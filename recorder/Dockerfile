FROM python:3.8.12-slim

COPY requirements.txt /app/recorder/
WORKDIR /app/recorder/
RUN apt-get update && \
    apt-get install libgl1-mesa-glx \
    'ffmpeg'\
    'libsm6'\
    'libxext6' -y

RUN pip install -r requirements.txt

COPY src/ ./

CMD [ "python", "-u", "capture.py" ]
