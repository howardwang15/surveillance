

# Features

- Connect to camera/record video
- Analyze video content/motion...detect object (people/animals)
- Handle recording crashes
- DB - store recorded clip of motion, start/end timestamps...retrieve
- Server to handle requests/queries
- Remove videos older than 1 month
- Email system - send email on first frame of motion detected...send video clip
- Webpage: playback video, controls, timeline


## Video storage and Retrieval

### GCP Storage

- up to 5 GB per month
- store mp4 files

### MySQL DB

**Schema**

- uuid
- start time (datetime)
- end time (datetime)
- file name (varchar)
- contains person (boolean)


### Client

- fetch videos from GCP and cache
- add controls to playback/seek videos
- send request to server for videos of interest

### Server

All timestamps are in Linux format (seconds since Epoch)

- GET `videos/`
  - Retrieve all videos
- GET `videos/?timestamp`
  - Retrieve all videos before `timestamp`
- GET `videos/?timestamp1-timestamp2`
  - Retrieve all videos between `timestamp1` and `timestamp2`

### Others

- Create job that automatically deletes videos older than 1 month (perhaps MySQL job)

