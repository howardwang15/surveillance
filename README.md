# Home surveillance system

## Features

- Connect to RSTP camera to record video
- Detects people, cats, dogs within video (using YOLO model), then sends email notification with captured frame and video
- Web server for accessing recordings

## Services

- recording service: captures video and runs object detection algorithm
- HTTP server: sends email notifications and renders web algorithm
- MySQL database: persists detections and image/video file locations with timestamps

Uses docker-compose to run all services 

## Configuration
To set-up the recording service, create JSON files like the following:
```JSON
{
    "link": "RTSP camera link w/ IP address",
    "camera_id": "Camera ID",
    "camera_name": "Camera name for camera",
    "min_detection_height": "minimum height of detected object for filtering (float between 0 and 1)",
    "min_hw_ratio": "minimum height to width ratio of detected object for filtering (float)"
}
```

### Environment Variables
- `CONFIG_FILE`: path to JSON config file relative to the capture script
- `WEIGHTS_FILE`: path to YOLO weights file relative to the capture script
- `EMAIL_SOURCE`: email address to send email from
- `EMAIL_DEST`: Python list of email addresses to send notification email to
- `EMAIL_PASS`: base64 encoded password to `EMAIL_SOURCE` email address
- `MYSQL_USER`: username to access MySQL database (set to `root`)
- `MYSQL_ROOT_PASSWORD`: root password to MySQL database


## Development

### Recording service

1. Create a virtual environment and activate
2. Run `pip install -r requirements.txt` under `recorder/`
3. Set `APP_ENV` environment variable to `development`. This disables writes into the MySQL database.
4. Run `python -u capture.py` in the `recorder/src` directory.

### Web server
1. Run `go get` under `server/`
2. Start the web server by running `go run app.go`
