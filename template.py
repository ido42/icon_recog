import numpy as np
import cv2
import os
import math
# here we will use the layout of the display to determine the regions of the real display,
# like the borders of the buttons, and screen from capacitive part
layout_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1_display_real_cropped.png")).replace('\\', '/')
l_out = cv2.imread(layout_dir)  # model image is loaded only for control purposes
real_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "ex1.jpeg")).replace('\\', '/')
real_img = cv2.imread(real_img_dir)
cv2.imshow("lay",l_out)
cv2.waitKey(0)
def match_display_template(img, layout=l_out):
    # images in black and white
    if len(img.shape) == 3:  # if the image is not already bw, convert
        if img.shape[2] == 3:  # sometimes it looks like there are 3 elemnts of .shape but the last is 1, thus bw
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(layout.shape) == 3:
        if layout.shape[2] == 3:
            layout = cv2.cvtColor(layout, cv2.COLOR_BGR2GRAY)
    # remove noise
    img = cv2.GaussianBlur(img, (7, 7), 0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
    #sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y
    canny_img=cv2.Canny(img,50,150)
    #sobx_lay = cv2.Sobel(layout, cv2.CV_64F, 1, 0, ksize=5)  # x
    #soby_lay = cv2.Sobel(layout, cv2.CV_64F, 0, 1, ksize=5)  # x
    canny_lay=cv2.Canny(layout,50,150)
    cv2.imshow("img",canny_img)
    cv2.imshow("lay",canny_lay)
    cv2.waitKey(0)

    lines = cv2.HoughLines(canny_lay, 1, np.pi / 180, 110 )

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # x1 = (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))
            # y1 = (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))
            # x2 = (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))
            # y2 = (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))

            x1 = max(0, x1)
            x2 = max(0, x2)
            y1 = max(0, y1)
            y2 = max(0, y2)

            x1 = min(layout.shape[1], x1)
            x2 = min(layout.shape[1], x2)
            y1 = min(layout.shape[0], y1)
            y2 = min(layout.shape[0], y2)
            pt1=(x1,y1)
            pt2=(x2,y2)
            cv2.line(layout, pt1, pt2, 255, 3, cv2.LINE_AA)
            cv2.imshow('linesDetected.jpg', layout)
            cv2.waitKey(0)
    # The below for loop runs till r and theta values
    # are in the range of the 2d array

"""    for r, theta in lines[0]:
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))
        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(layout, (x1, y1), (x2, y2), (0, 0, 255), 2)"""

    # img and

match_display_template(real_img,l_out)