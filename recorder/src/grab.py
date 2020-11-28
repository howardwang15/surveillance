import numpy as np
import datetime
import cv2
import os
from dotenv import load_dotenv


def main():
    load_dotenv()
    cap = cv2.VideoCapture(os.getenv('CAMERA_URL'))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    i = 0
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        fgmask = fgbg.apply(frame)
        cnt = np.count_nonzero(fgmask)
        print(cnt)
        if cnt * 25 > frame.shape[0] * frame.shape[1]:
            print('motion: {}'.format(i))
            i += 1

        cv2.imshow('fgmask', fgmask)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()