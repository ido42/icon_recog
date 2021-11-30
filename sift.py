import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
"""
def display(match):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(match)
    plt.show()

# directories of the images
model_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1.jpg")).replace('\\', '/')
snowflake_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "buttons", "timer.png")).replace('\\', '/')
real_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1_display_real_cropped.jpeg")).replace('\\', '/')

# upload images to matrices
model = cv2.imread(model_img_dir)  # model image is loaded only for control purposes
real_img = cv2.imread(real_img_dir)
sf = cv2.imread(snowflake_img_dir)

# scale the real image to the image of the model image (0.36 is their approx. ratio)
scale_percent = 0.36 # percent of original size (works in range 0.26 to 1.5)
width = int(real_img.shape[1] * scale_percent)
height = int(real_img.shape[0] * scale_percent)
dim = (width, height)
real_img = cv2.resize(real_img, dim, interpolation=cv2.INTER_AREA)

# images in black and white
model_cp = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
real_cp = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
sf_cp = cv2.cvtColor(sf, cv2.COLOR_BGR2GRAY)
"""

def sift_matching(display_img, icon):
    # images in black and white
    if len(display_img.shape) == 3:  # if the image is not already bw, convert
        if display_img.shape[2] == 3:  # sometimes it looks like there are 3 elemnts of .shape but the last is 1, thus bw
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    if len(icon.shape) == 3:
        if icon.shape[2] == 3:
            icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)

    # first gaussian blur then canny edge detection on the images for reducing noise and focusing on edges
    display_gs = cv2.GaussianBlur(display_img, (3, 3), cv2.BORDER_DEFAULT)
    display_canny=cv2.Canny(display_gs, 40, 100)
    icon_gs = cv2.GaussianBlur(icon, (3, 3), cv2.BORDER_DEFAULT)
    icon_canny=cv2.Canny(icon_gs, 40, 100)

    # display images (just control)
#    cv2.imshow("real!", display_img)
#    cv2.imshow("sf!", icon)

#    cv2.imshow("real c", display_canny)
#    cv2.imshow("sf c", icon_canny)
#    cv2.waitKey((0))

    sift = cv2.SIFT_create()  # needs extra opencv-contrib files (solved)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    kp1_sift, des1_sift = sift.detectAndCompute(icon_canny, None)
    kp2_sift, des2_sift = sift.detectAndCompute(display_canny, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1_sift,des2_sift,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img3 = cv2.drawMatchesKnn(icon,kp1_sift,display_img,kp2_sift,matches,None,**draw_params)
   # cv2.imshow("",img3)
   # cv2.waitKey(0)
    return img3
#    plt.imshow(img3,),plt.show()
