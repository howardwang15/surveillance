FROM python:3

COPY requirements.txt /app/server/
WORKDIR /app/server

RUN pip install -r requirements.txt

COPY . /app/server

RUN ls -la

CMD [ "flask", "run", "--host=0.0.0.0" ]
