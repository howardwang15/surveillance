#streaming Focus68
import numpy as np
import datetime
import cv2
import os
import argparse
import tensorflow as tf
import mysql.connector
import base64
import json
import logging
import sys
from logger import Logger
from models import YoloV3Tiny, YoloV3
from email_service import EmailService
from image_operator import ImageOperator
from dotenv import load_dotenv
from multiprocessing import Process


def get_options():
    """Add and parse command line arguments

    Returns:
        argparse.Namespace: options parsed while running program
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', help='path to YOLO weights')
    parser.add_argument('-c', '--config', help='path to configuration file')
    options = parser.parse_args()
    return options


class Recorder():
    def __init__(self, weights, config_file, tiny, buffer_length=150):
        with open(config_file) as f:
            self.config = json.load(f)

        self.link = self.config['link']
        self.camera_id = self.config['camera_id']
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.buffer_length = buffer_length
        self.video_buffer = [None] * self.buffer_length
        self.yolo_model = YoloV3Tiny(classes=80) if tiny else YoloV3(classes=80)
        self.yolo_model.load_weights(weights)
        self.cap = cv2.VideoCapture(self.link)
        self.logger = Logger(
            name='recorder logger',
            log_path='./logs/capture.log',
            default_level=logging.DEBUG,
            max_size=1024*1024*3,
            num_files=5
        )
        self.image_operator = ImageOperator(
            config=self.config,
            logger=self.logger
        )

    def capture(self):
        self.logger.write(logging.INFO, 'Initializing...')
        self.logger.write(logging.INFO, self.config)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        frame_pos = 0
        in_record = 0
        video_name = ''
        if not os.path.exists(os.path.join('..', 'files')):
            os.makedirs(os.path.join('..', 'files'))

        # get the coco class names and the classes we're interested in
        class_names = np.array([c.strip() for c in open('coco.names').readlines()])

        targets = [0, 16, 20]
        startup = True
        c = 0

        while True:
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.link)
                self.logger.write(logging.INFO, 'opening cap again')
                continue

            # capture frame...
            ret, frame = self.cap.read()

            if not ret:
                self.logger.write(logging.ERROR, 'Couldn\'t get frame')
                self.logger.write(logging.INFO, 'cap is opened: {}'.format(self.cap.isOpened()))
                self.cap.release()
                self.cap = cv2.VideoCapture(self.link)
                continue

            c += 1
            if c % 1000 == 0:
                self.logger.write(logging.DEBUG, c)


            # resize, store into buffer
            frame_process = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            h_process, w_process, _ = frame_process.shape
            h_full, w_full, _ = frame.shape
            frame_pos = 0 if frame_pos + 1 >= self.buffer_length else frame_pos + 1
            self.video_buffer[frame_pos] = frame
            

            # hack for making sure service doesn't crash if there's motion right when it starts up
            if startup:
                self.video_buffer = [frame for _ in range(self.buffer_length)]
                startup = False

            # apply filter to get foreground
            fgmask = fgbg.apply(frame_process)
            
            if in_record > 0:
                in_record += 1
        
                # finish writing to file
                if in_record >= self.buffer_length:
                    in_record = 0
                    out.release()
                    self.email_service = EmailService(
                        os.getenv('EMAIL_SOURCE'),
                        ', '.join(json.loads(os.getenv('EMAIL_DEST'))),
                        base64.b64decode(os.getenv('EMAIL_PASS')).decode()
                    )

                    Process(
                        target=self.email_service.send_email,
                        args=(os.path.join('..', 'files', frame_name), os.path.join('..', 'files', video_name), now)
                    ).start()

            cnt = np.count_nonzero(fgmask)

            # threshold...filters out images with too much noise
            if cnt * 25 > h_process * w_process:
                # see if person (class 0) is in predictions
                if in_record == 0:
                    self.logger.write(logging.INFO, 'motion detected')

                    # process image and run through model
                    tf_frame = self.image_operator.transform_images(tf.expand_dims(cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB), 0))
                    boxes, scores, classes, nums = self.yolo_model.predict(tf_frame)

                    # filter the detections that are people and with confidence level > 0.56
                    num_out, classes, detected_classes, detected_boxes, detected_scores = self.image_operator.filter_detections(
                        nums,
                        classes,
                        scores,
                        boxes,
                        targets
                    )

                    if num_out > 0:
                        # start to save frames to video
                        self.logger.write(logging.INFO, 'Detected objects of interest: {}'.format(class_names[detected_classes]))
                        in_record = 1
                        start_pos = frame_pos - 30
                        ts = datetime.datetime.now().timestamp()

                        # get timestamp and create file names
                        readable = datetime.datetime.fromtimestamp(ts).isoformat()
                        self.logger.write(logging.INFO, 'writing to new file: {}'.format(readable))
                        video_name = 'videos_{}_{}.mp4'.format(self.camera_id, readable)  # create new filename
                        frame_name = 'images_{}_{}.png'.format(self.camera_id, readable)
                        original_name = 'original_{}_{}.png'.format(self.camera_id, readable)

                        # insert alert into database
                        now = datetime.datetime.now().isoformat()
                        insert_video_query = "INSERT INTO videos (start_time, video_name, first_frame) values ('{}', '{}', '{}')".format(now, video_name, frame_name)
                        self.cnx = mysql.connector.connect(user=os.getenv('MYSQL_USER'), password=os.getenv('MYSQL_ROOT_PASSWORD'), host='mysql', database='surveillance')
                        self.cursor = self.cnx.cursor()
                        self.cursor.execute(insert_video_query)
                        self.cnx.commit()
                        video_id = self.cursor.lastrowid

                        # add detected objects to database
                        for _class in detected_classes:
                            class_name = class_names[int(_class)]
                            insert_object_query = "INSERT INTO detections (video_id, type) values ('{}', '{}')".format(video_id, class_name)
                            self.cursor.execute(insert_object_query)
                            self.cnx.commit()

                        # draw bounding boxes and create video writer
                        out = cv2.VideoWriter(os.path.join('..', 'files', video_name), self.fourcc, 15.0, (w_full, h_full))
                        cv2.imwrite(os.path.join('..', 'files', original_name), frame)
                        frame = self.image_operator.draw_outputs(frame,
                            (detected_boxes, detected_scores, detected_classes, num_out),
                            class_names
                        )

                        # save captured frame
                        cv2.imwrite(os.path.join('..', 'files', frame_name), frame)
                    else:
                        self.logger.write(logging.INFO, 'found objects not interested in: {}'.format(class_names[classes]))

            if in_record > 0:
                out.write(self.video_buffer[frame_pos - 30])  # write frame
                
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    load_dotenv()
    options = get_options()
    recorder = Recorder(weights=options.weights, config_file=options.config, tiny=False)
    recorder.capture()
