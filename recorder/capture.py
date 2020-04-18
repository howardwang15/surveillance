#streaming Focus68
import numpy as np
import datetime
import cv2
import os
import argparse
import tensorflow as tf
import mysql.connector
import base64
from models import YoloV3Tiny, YoloV3
from email_service import EmailService
from dotenv import load_dotenv
from google.cloud import storage


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', help='path to YOLO weights')
    options = parser.parse_args()
    return options

def transform_images(x_train, size=416):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

class Recorder():
    def __init__(self, weights, tiny, buffer_length=150):
        self.link = os.getenv('CAMERA_URL')
        self.camera_id = os.getenv('CAMERA_ID')
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.buffer_length = buffer_length
        self.video_buffer = [None] * self.buffer_length
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(os.getenv('BUCKET'))
        self.yolo_model = YoloV3Tiny(classes=80) if tiny else YoloV3(classes=80)
        self.yolo_model.load_weights(weights)
        self.cnx = mysql.connector.connect(user=os.getenv('MYSQL_USER'), password=os.getenv('MYSQL_ROOT_PASSWORD'), host='mysql', database='surveillance')
        self.cursor = self.cnx.cursor()
        self.email_service = EmailService(os.getenv('EMAIL_SOURCE'), os.getenv('EMAIL_DEST'), base64.b64decode(os.getenv('EMAIL_PASS')).decode())
    
    def capture(self):
        print('Initializing...')
        cap = cv2.VideoCapture(self.link)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        frame_pos = 0
        in_record = 0
        video_name = ''
        if not os.path.exists(os.path.join('..', 'files')):
            os.makedirs(os.path.join('..', 'files'))

        class_names = [c.strip() for c in open('coco.names').readlines()]

        while True:
            print(in_record)
            if not cap.isOpened():
                cap.open(self.link)
                continue

            # capture frame...
            ret, frame = cap.read()

            if frame is None or frame.size == 0:
                continue

            # resize, store into buffer
            frame_resize = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frame_process = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            h_process, w_process, _ = frame_process.shape
            h_full, w_full, _ = frame_resize.shape
            frame_pos = 0 if frame_pos + 1 >= self.buffer_length else frame_pos + 1
            self.video_buffer[frame_pos] = frame

            # apply filter to get foreground
            fgmask = fgbg.apply(frame_process)

            # find contours...for debugging purposes
            contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 500:  # filter contours that are too small
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                x *= 4
                y *= 4
                w *= 4
                h *= 4
            
            if in_record > 0:
                in_record += 1
        
                # finish writing to file
                if in_record >= self.buffer_length:
                    in_record = 0
                    out.release()
                    blob = self.bucket.blob(video_name)
                    self.email_service.send_email(os.path.join('..', 'files', frame_name), os.path.join('..', 'files', video_name), now)
                    # blob.upload_from_filename(os.path.join('files', video_name))  

            cnt = np.count_nonzero(fgmask)
            # threshold...filters out images with too much noise
            if cnt * 20 > h_process * w_process:
                # see if person (class 0) is in predictions
                if in_record == 0:
                    tf_frame = transform_images(tf.expand_dims(cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB), 0))
                    boxes, scores, classes, nums = self.yolo_model.predict(tf_frame)

                    # filter the detections that are people
                    filter_indices = np.nonzero(np.isin(classes[0][:nums[0]], [0, 2]))
                    num_out = filter_indices[0].shape[0]
                    classes = classes[0][:nums[0]][filter_indices]
                    boxes = boxes[0][:nums[0]][filter_indices]
                    scores = scores[0][:nums[0]][filter_indices]

                    if np.any(np.in1d([0, 16, 17, 18, 19, 20], classes)):
                        in_record = 1
                        start_pos = frame_pos - 30
                        ts = datetime.datetime.now().timestamp()

                        readable = datetime.datetime.fromtimestamp(ts).isoformat()
                        print('writing to new file: {}'.format(readable))
                        video_name = 'videos_{}_{}.mp4'.format(self.camera_id, readable)  # create new filename
                        frame_name = 'images_{}_{}.png'.format(self.camera_id, readable)

                        now = datetime.datetime.now().isoformat()
                        insert_video_query = "INSERT INTO videos (start_time, video_name, first_frame) values ('{}', '{}', '{}')".format(now, video_name, frame_name)

                        self.cursor.execute(insert_video_query)
                        self.cnx.commit()
                        video_id = self.cursor.lastrowid

                        for _class in classes:
                            class_name = class_names[int(_class)]
                            insert_object_query = "INSERT INTO detections (video_id, type) values ('{}', '{}')".format(video_id, class_name)
                            self.cursor.execute(insert_object_query)
                            self.cnx.commit()

                        out = cv2.VideoWriter(os.path.join('..', 'files', video_name), self.fourcc, 15.0, (w_process, h_process))
                        frame = draw_outputs(frame, (boxes, scores, classes, num_out), class_names)

                        cv2.imwrite(os.path.join('..', 'files', frame_name), frame)

            if in_record > 0:
                out.write(frame_process)  # write frame
                
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    load_dotenv()
    options = get_options()
    recorder = Recorder(weights=options.weights, tiny=False)
    recorder.capture()
