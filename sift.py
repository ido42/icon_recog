import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans
# here is te sift matching algorithm in the form of a function,
# it outputs the matching points for both the model and the image in the form of two lists
# it also returns the matching image


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
    display_canny=cv2.Canny(display_gs,30, 150)
    icon_gs = cv2.GaussianBlur(icon, (3, 3), cv2.BORDER_DEFAULT)
    icon_canny=cv2.Canny(icon_gs, 30, 150)

    # display images (just control)
#    cv2.imshow("real!", display_img)
#    cv2.imshow("sf!", icon)

    #cv2.imshow("real c", display_canny)
    #cv2.imshow("sf c", icon_canny)
    #cv2.waitKey((0))

    sift = cv2.SIFT_create()  # needs extra opencv-contrib files (solved)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    kp1_sift, des1_sift = sift.detectAndCompute(icon_canny, None)
    kp2_sift, des2_sift = sift.detectAndCompute(display_canny, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_prm = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_prm = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_prm,search_prm)
    matches = flann.knnMatch(des1_sift,des2_sift,k=2)
    #print(matches[0][0].distance)
    # only good matches,  mask bad matches
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_prm = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    # index lists for matches 1:icon 2:image
    l_kp1 = []
    l_kp2 = []

    for mtc in range(len(matches)):
        # Get the matching keypoints for each of the images
        if matchesMask[mtc][0]==1:
            idx1 = matches[mtc][0].queryIdx
            idx2 = matches[mtc][0].trainIdx

            # Get the coordinates x:col, y:row
            (x1, y1) = int(kp1_sift[idx1].pt[0]),int(kp1_sift[idx1].pt[1])
            (x2, y2) = int(kp2_sift[idx2].pt[0]),int(kp2_sift[idx2].pt[1])

            l_kp1.append((x1, y1))
            l_kp2.append((x2, y2))

    img3 = cv2.drawMatchesKnn(icon,kp1_sift,display_img,kp2_sift,matches,None,**draw_prm)
    #cv2.imshow("",img3)
   # cv2.waitKey(0)
    return img3,l_kp1,l_kp2
#    plt.imshow(img3,),plt.show()
def cluster_screen(img):
    x=[]
    #img=cv2.normalize(img,0,255,cv2.NORM_MINMAX)
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            x.append([j,i,img[j,i]])

    kmeans = KMeans(2)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)

    for t in range(img.size):
        if identified_clusters[t]==0:
            cv2.circle(img, x[t][0:2], 3, 0, -1)
        else:
            cv2.circle(img, x[t][0:2], 3, 255, -1)
    return img