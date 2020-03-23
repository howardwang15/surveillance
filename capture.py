#streaming Focus68
import numpy as np
import datetime;
import cv2
import os

link='rtsp://user:pass@192.168.0.103:6667/blinkhd'
camera_id = 'front_door'
key=0
dir = './videos/'+camera_id+'/'
cnt = 0
video_buffer = [None] * 150
length = 150
cur_pos = 0
start_pos = -10
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

cap = cv2.VideoCapture(link)
ret, frame = cap.read()


frame_pre=cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(frame_pre)
in_record = 0


try:
    os.stat(dir)
except:
    os.mkdir(dir)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape)
    if frame is None:
        print('.')
        cap = cv2.VideoCapture(link)
        continue


    frame_1=cv2.resize(frame,(0,0), fx=0.5, fy=0.5)
    frame_cur=cv2.resize(frame_1,(0,0), fx=0.5, fy=0.5)
    m, n, _ = frame_cur.shape
    m1, n1, _ = frame.shape
    video_buffer[cur_pos] = frame_1
    cur_pos += 1
    if cur_pos >= length:
        cur_pos = 0

    fgmask = fgbg.apply(frame_cur)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #(im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # looping for contours
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue

        # get bounding box from countour
        (x, y, w, h) = cv2.boundingRect(c)
        x *=4
        y *=4
        w *=4
        h *=4

        # draw bounding box
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if in_record > 0:
        print(in_record)

    if in_record > 0:
        in_record += 1
        if in_record >= length:
            in_record = 0
            out.release()
    cv2.imshow('foreground and background', fgmask)
    #cv2.imshow('rgb', frame_1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    #print diff
    cnt=np.count_nonzero(fgmask)
    if cnt *20>m*n:
        if in_record == 0:
            in_record = 1
            start_pos = cur_pos - 30

            ts = datetime.datetime.now().timestamp()

            readable = datetime.datetime.fromtimestamp(ts).isoformat()
            print(readable)
            video_name = dir + readable + '.mp4'
            out = cv2.VideoWriter(video_name, fourcc, 15.0, (n1, m1))
    if in_record > 0:
        out.write(frame)
    frame_pre=frame_cur.copy()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
