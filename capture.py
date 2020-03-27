#streaming Focus68
import numpy as np
import datetime
import cv2
import os
import argparse
import tensorflow as tf
from models import YoloV3Tiny, YoloV3
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
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
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
        self.yolo_model = YoloV3Tiny(classes=80) if tiny else YoloV3Tiny(classes=80)
        self.yolo_model.load_weights(weights)
    
    def capture(self):
        print('Initializing...')
        cap = cv2.VideoCapture(self.link)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        frame_pos = 0
        in_record = 0
        video_name = ''
        if not os.path.exists('files'):
            os.makedirs('files')

        class_names = [c.strip() for c in open('coco.names').readlines()]

        while True:
            if not cap.isOpened():
                cap.open(self.link)
                continue
            
            # capture frame...
            ret, frame = cap.read()
            
            if frame.size == 0:
                continue

            # resize, store into buffer
            frame_process = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            h_process, w_process, _ = frame_process.shape
            h_full, w_full, _ = frame.shape
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
                    # blob.upload_from_filename(os.path.join('files', video_name))  

            cnt = np.count_nonzero(fgmask)
            # threshold...filters out images with too much noise
            if cnt * 200 > h_process * w_process:
                tf_frame = transform_images(tf.expand_dims(cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB), 0))
                boxes, scores, classes, nums = self.yolo_model.predict(tf_frame)
                

                # see if person (class 0) is in predictions
                if in_record == 0 and 0 in classes[0][:nums[0]]:
                    in_record = 1
                    start_pos = frame_pos - 30
                    ts = datetime.datetime.now().timestamp()

                    readable = datetime.datetime.fromtimestamp(ts).isoformat()
                    print('writing to new file: {}'.format(readable))
                    video_name = 'videos_{}_{}.mp4'.format(self.camera_id, readable)  # create new filename
                    frame_name = 'images_{}_{}.png'.format(self.camera_id, readable)
                    out = cv2.VideoWriter(os.path.join('files', video_name), self.fourcc, 15.0, (w_full, h_full))
                    frame = draw_outputs(frame, (boxes, scores, classes, nums), class_names)
                    
                    cv2.imwrite(os.path.join('files', frame_name), frame)

            if in_record > 0:
                out.write(frame)  # write frame
                
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    load_dotenv()
    options = get_options()
    recorder = Recorder(weights=options.weights, tiny=True)
    recorder.capture()
