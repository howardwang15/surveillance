#streaming Focus68
import numpy as np
import datetime
import cv2
import os
from dotenv import load_dotenv
from google.cloud import storage


class Recorder():
    def __init__(self, buffer_length=150):
        self.link = os.getenv('CAMERA_URL')
        self.camera_id = os.getenv('CAMERA_ID')
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.buffer_length = buffer_length
        self.video_buffer = [None] * self.buffer_length
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(os.getenv('BUCKET'))
    
    def capture(self):
        print('Initializing...')
        cap = cv2.VideoCapture(self.link)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        frame_pos = 0
        in_record = 0
        video_name = ''
        while True:
            if not cap.isOpened():
                cap.open(self.link)
                continue
            
            ret, frame = cap.read()

            frame_process = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            h_process, w_process, _ = frame_process.shape
            h_full, w_full, _ = frame.shape
            frame_pos = 0 if frame_pos + 1 >= self.buffer_length else frame_pos + 1
            self.video_buffer[frame_pos] = frame
            fgmask = fgbg.apply(frame_process)

            contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                x *= 4
                y *= 4
                w *= 4
                h *= 4
            
            if in_record > 0:
                in_record += 1
        
                if in_record >= self.buffer_length:
                    in_record = 0
                    out.release()
                    blob = self.bucket.blob(video_name)
                    blob.upload_from_filename(video_name)

            cnt = np.count_nonzero(fgmask)
            if cnt * 20 > h_process * w_process:
                if in_record == 0:
                    in_record = 1
                    start_pos = frame_pos - 30
                    ts = datetime.datetime.now().timestamp()

                    readable = datetime.datetime.fromtimestamp(ts).isoformat()
                    print('writing to new file: {}'.format(readable))
                    video_name = 'videos_{}_{}.mp4'.format(self.camera_id, readable)
                    out = cv2.VideoWriter(video_name, self.fourcc, 15.0, (w_full, h_full))

            if in_record > 0:
                out.write(frame)
                
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    load_dotenv()
    recorder = Recorder()
    recorder.capture()
