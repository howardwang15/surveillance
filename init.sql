-- Use these queries to setup new surveillance database

CREATE DATABASE surveillance;

USE surveillance;

CREATE TABLE videos (
    id int NOT NULL AUTO_INCREMENT,
    start_time datetime NOT NULL,
    video_name text NOT NULL,
    first_frame text NOT NULL,
    camera_id text NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE detections (
    id int NOT NULL AUTO_INCREMENT,
    video_id int NOT NULL,
    type text NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

SHOW TABLES;  -- verify that tables were created
