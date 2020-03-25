FROM python:3

COPY requirements.txt /app/
WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app/

RUN ls -la

CMD [ "python", "-u", "capture.py", "--weights=weights/yolov3-tiny.tf" ]