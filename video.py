import numpy as np
import cv2 as cv
import os
from sift import *
from sklearn.cluster import KMeans

# here everything so far works in the form of video, yay!
video_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "try.mp4")).replace('\\', '/')
snowflake_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "buttons", "snow_flake.png")).replace('\\', '/')
eco_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "buttons", "timer.png")).replace('\\', '/')
wifi_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "icons", "wifi_ir.jpeg")).replace('\\', '/')

sf = cv2.imread(snowflake_dir)
eco = cv2.imread(eco_dir)
wifi = cv2.imread(wifi_dir)
icon_list = [sf, eco, wifi]

eco_tr=0
wifi_tr=0
sf_tr=0

temp_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1template.png")).replace('\\', '/')
temp = cv2.imread(temp_img_dir)
t_cp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
t_gs = cv2.GaussianBlur(t_cp, (5, 5), cv2.BORDER_DEFAULT)
t_canny = cv2.Canny(t_gs, 50, 100)
cap = cv.VideoCapture(video_dir)
eco_state = "off"
wifi_state = "off"
i = 0  # index
idil = 0
while cap.isOpened():
    ret, frame = cap.read()
    # frame is read-> ret is True
    if not ret:
        print("Error")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)



    i_gs = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    i_canny = cv2.Canny(i_gs, 50, 100)
    cv2.imshow("img", i_canny)

    matched, l1, l2=sift_matching(gray, eco)
    matched_w, w1, w2 = sift_matching(gray, wifi)

    # find the display
    res = cv2.matchTemplate(i_canny, t_canny, cv2.TM_CCOEFF)
    cv2.imshow("heat map", res)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    row_temp, col_temp = t_cp.shape
    bottom_right_loc = (max_loc[0] + col_temp, max_loc[1] + row_temp)
    cv2.rectangle(gray, max_loc, bottom_right_loc, 255, 5)
    p_l1 = (int(max_loc[0]), int(max_loc[1] + row_temp * 4 / 7))
    p_l2 = (int(max_loc[0] + col_temp), int(max_loc[1] + row_temp * 4 / 7))
    cv2.line(gray, p_l1, p_l2, 255, 3)

    kmeans = KMeans(2)
    x = l2
    frame_cp = frame
    if len(l2) == 1:
        if l2[0][1] < p_l1[1]:
            # eco_state = "on"
            eco_tr += 1
        # else:
        #   eco_state = "off"
    try:
        kmeans.fit(x)
        identified_clusters = kmeans.fit_predict(x)
        red = (0, 0, 255)
        green = (0, 255, 0)
        cv.circle(frame_cp, (np.int(kmeans.cluster_centers_[0][0]), np.int(kmeans.cluster_centers_[0][1])), 3, red, -1)
        cv.circle(frame_cp, (np.int(kmeans.cluster_centers_[1][0]), np.int(kmeans.cluster_centers_[1][1])), 3, green,
                  -1)
        if np.any(kmeans.cluster_centers_[:, 1] < p_l1[1]):
            # eco_state = "on"
            eco_tr += 1
        # else:
        #   eco_state = "off"
    except:
        pass
    kmeans2 = KMeans(2)
    x = w2
    #frame_cp = frame
    if len(w2) == 1:
        if w2[0][1] < p_l1[1]:
            # eco_state = "on"
            eco_tr += 1
        # else:
        #   eco_state = "off"
    try:
        kmeans2.fit(w2)
        identified_clusters2 = kmeans2.fit_predict(w2)
        cv.circle(frame_cp, (np.int(kmeans2.cluster_centers_[0][0]), np.int(kmeans2.cluster_centers_[0][1])), 3, red, -1)
        cv.circle(frame_cp, (np.int(kmeans2.cluster_centers_[1][0]), np.int(kmeans2.cluster_centers_[1][1])), 3, green,
                  -1)
        if np.any(kmeans2.cluster_centers_[:, 1] < p_l1[1]):
            # eco_state = "on"
            wifi_tr += 1
        # else:
        #   eco_state = "off"
    except:
        pass
    i += 1
    if i==12:
        i=0
        if eco_tr>6:
            eco_state="on"
        else:
            eco_state="off"
        if wifi_tr>4:
            wifi_state="on"
        else:
            wifi_state="off"
        eco_tr=0
        wifi_tr=0
    cv2.putText(
        matched,  # numpy array on which text is written
        "Eco-mod: ",  # text
        (0,200),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        0.35,  # font size
        (209, 80, 209, 255),  # font color
        1)  # font stroke
    cv2.putText(
        matched,  # numpy array on which text is written
         eco_state,  # text
        (0, 220),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        0.35,  # font size
        (209, 80, 209, 255),  # font color
        1)  # font stroke
    cv2.putText(
        matched,  # numpy array on which text is written
        "wifi: ",  # text
        (0, 240),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        0.35,  # font size
        (209, 80, 209, 255),  # font color
        1)  # font stroke
    cv2.putText(
        matched,  # numpy array on which text is written
        wifi_state,  # text
        (0, 260),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        0.35,  # font size
        (209, 80, 209, 255),  # font color
        1)  # font stroke
    cv.imshow("clustered", frame_cp)
    cv.imshow('frame', matched)
    #print(eco_state)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()