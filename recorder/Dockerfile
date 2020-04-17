FROM python:3

COPY requirements.txt /app/recorder/
WORKDIR /app/recorder

RUN pip install -r requirements.txt

COPY . /app/recorder

RUN ls -la


CMD [ "python", "-u", "capture.py", "--weights=weights/yolov3.tf" ]
