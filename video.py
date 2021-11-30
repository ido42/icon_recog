import numpy as np
import cv2 as cv
import os
from sift import *
video_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "display touched.mp4")).replace('\\', '/')
snowflake_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "buttons", "snowflake_umb3.jpg")).replace('\\', '/')
sf = cv2.imread(snowflake_img_dir)

cap = cv.VideoCapture(video_dir)
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    matched=sift_matching(gray, sf)
    cv.imshow('frame', matched)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()