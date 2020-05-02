# Recorder service

## Entering container

Run `sudo docker ps`...output should be 3 rows

- find row with `IMAGE=surveillance_server` and copy the container ID

- run the command `sudo docker exec -it <ID> bash`. This should take you inside the container

## Environment variables

- environment variables are located in the .env file in the root directory
