FROM python:3

WORKDIR /app
COPY . /app/

RUN ls -la
RUN pip install -r requirements.txt

CMD [ "python", "-u", "capture.py" ]