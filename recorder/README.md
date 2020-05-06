# Recorder service

## Testing without Docker

- make sure `test_cap.py` is the same as `capture.py`...remove the mysql stuff

- activate virtual environment by running `source recorder_env/bin/activate`

- run `python3 test_cap.py --weights=weights/yolov3.tf` to start the script

## Entering container

Run `sudo docker ps`...output should be 3 rows

- find row with `IMAGE=surveillance_recorder` and copy the container ID

- run the command `sudo docker exec -it <ID> bash`. This should take you inside the container

- to exit, use CTRL-D

## Environment variables

- environment variables are located in the .env file in the root directory
