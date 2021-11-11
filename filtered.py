import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def display(match):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(match)
    plt.show()
snowflake_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "snow_flake.jpg")).replace('\\', '/')
umb3_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "snowflake_umb3.jpg")).replace('\\', '/')
lower_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "lower.PNG")).replace('\\', '/')
timer_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "timer.png")).replace('\\', '/')
upper_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "upper.jpg")).replace('\\', '/')
wifi_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "wifi.jpg")).replace('\\', '/')

real_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "wifi on cropped.jpeg")).replace('\\', '/')
model_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1.jpg")).replace('\\', '/')
model = cv2.imread(model_img_dir)
real_img = cv2.imread(real_img_dir)
icons_dir=(snowflake_img_dir,umb3_dir,lower_dir,timer_dir,upper_dir,wifi_dir)
icons_name=("frost","vacation","lower","timer","upper","wifi")

scale_percent = 0.36 # percent of original size

width = int(real_img.shape[1] * scale_percent)
height = int(real_img.shape[0] * scale_percent)
dim = (width, height)
real_img = cv2.resize(real_img, dim, interpolation=cv2.INTER_AREA)


cv2.imshow("found!", real_img)
cv2.waitKey((0))
model_cp = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
real_cp = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
real_gs = cv2.GaussianBlur(real_cp, (5, 5), cv2.BORDER_DEFAULT)

real_canny=cv2.Canny(real_gs, 100, 200)
for icon in range(len(icons_dir)):
    ic = cv2.imread(icons_dir[icon])
    ic_cp = cv2.cvtColor(ic, cv2.COLOR_BGR2GRAY)
    ic_gs = cv2.GaussianBlur(ic_cp, (5, 5), cv2.BORDER_DEFAULT)
    ic_canny=cv2.Canny(ic_gs, 100, 200)
    res = cv2.matchTemplate(model_cp, ic_cp, cv2.TM_CCOEFF)
    cv2.imshow("heat map", res)
    cv2.waitKey(0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    row_temp, col_temp = ic_cp.shape
    bottom_right_loc = (max_loc[0]+col_temp, max_loc[1]+row_temp)
    cv2.rectangle(real_img, max_loc, bottom_right_loc, (255, 0, 0), 5)
    position = (max_loc[0]+5, max_loc[1]-15)
    cv2.putText(
    real_img,  # numpy array on which text is written
    icons_name[icon],  # text
    position,  # position at which writing has to start
    cv2.FONT_HERSHEY_SIMPLEX,  # font family
    0.5,  # font size
    (209, 80, 0, 255),  # font color
    2)  # font stroke
    cv2.imshow("found!", real_img)
    cv2.waitKey((0))

wifi_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "wifi displ.png")).replace('\\', '/')
w = cv2.imread(wifi_dir)
w_cp = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY)
resw = cv2.matchTemplate(model_cp, w_cp, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resw)
row_temp, col_temp = w_cp.shape
bottom_right_loc = (max_loc[0]+col_temp, max_loc[1]+row_temp)
cropped_image = model_cp[max_loc[0]:bottom_right_loc[0], max_loc[1]:bottom_right_loc[1]]
cv2.imshow("w map", cropped_image)
cv2.waitKey(0)
cropped_real = real_cp[max_loc[0]:bottom_right_loc[0], max_loc[1]:bottom_right_loc[1]]
cv2.imshow("w map", cropped_real)
cv2.waitKey(0)

orb = cv2.ORB_create()
cropped_image = cv2.Canny(cropped_image, 100, 200)
cropped_real = cv2.Canny(cropped_real, 100, 200)

key_points1, descriptors1 = orb.detectAndCompute(cropped_real, None) # detecying key points and descriptors for model and template
key_points2, descriptors2 = orb.detectAndCompute(cropped_image, None)
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = brute_force.match(descriptors1, descriptors2)
if len(matches)==0:
    print("off")
else: print("on")
"""for icon in icons_dir:
    ic=cv2.imread(icon)

    ic_cp = cv2.cvtColor(ic, cv2.COLOR_BGR2GRAY)
    ic_gs = cv2.GaussianBlur(ic_cp, (5, 5), cv2.BORDER_DEFAULT)

    ic_canny = cv2.Canny(ic_cp, 100, 200)
    res = cv2.matchTemplate(real_cp, ic_canny, cv2.TM_CCOEFF)
    cv2.imshow("heat map", res)
    cv2.waitKey(0)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    row_temp, col_temp = ic_cp.shape
    bottom_right_loc = (max_loc[0] + col_temp, max_loc[1] + row_temp)
    cv2.rectangle(real_img, max_loc, bottom_right_loc, (255, 0, 0), 5)
    cv2.imshow("found!", real_img)
    cv2.waitKey(0)"""

#model_cp = cv2.GaussianBlur(model_cp, (3, 3), cv2.BORDER_DEFAULT)
#real_cp = cv2.GaussianBlur(real_cp, (3, 3 ), cv2.BORDER_DEFAULT)
"""kernel = np.ones((5,5), np.uint8)

model_canny=cv2.Canny(model_cp, 100, 200)
real_canny=cv2.Canny(real_cp, 100, 200)
model_canny = cv2.dilate(model_canny, kernel, iterations=1)
real_canny = cv2.dilate(real_canny, kernel, iterations=1)

cv2.imshow("model edge", model_canny)
cv2.imshow("real edge", real_canny)
cv2.waitKey(0)


orb = cv2.ORB_create()
key_points1, descriptors1 = orb.detectAndCompute(real_canny, None) # detecying key points and descriptors for model and template
key_points2, descriptors2 = orb.detectAndCompute(model_canny, None)
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = brute_force.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance) # distance indicates similarity, here we sort them from most similar to least
ss_match = cv2.drawMatches(real_cp, key_points1, model_cp, key_points2, matches[:5], None, flags=2)
display(ss_match)

# sift matching
sift = cv2.SIFT_create() # needs extra opencv-contrib files
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

kp1_sift, des1_sift = sift.detectAndCompute(real_canny, None)
kp2_sift, des2_sift = sift.detectAndCompute(model_canny, None)
#sift_matches = bf.knnMatch(des1_sift, des2_sift, k=2)
#sf_match = cv2.drawMatchesKnn(snowflake_cp, kp1_sift, model_cp, kp2_sift, sift_matches[:10], None, flags=2)
#display(sf_match)
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

img3 = cv2.drawMatchesKnn(real_canny,kp1_sift,model_canny,kp2_sift,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
"""
"""# straight forward template matching
rows, cols, channels = model.shape

# the methods that can be used are cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_SQDIFF etc.
res = cv2.matchTemplate(model_cp, snowflake_cp, cv2.TM_CCOEFF)
cv2.imshow("heat map", res)
cv2.waitKey(0)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
row_temp, col_temp = snowflake_cp.shape
bottom_right_loc = (max_loc[0]+col_temp, max_loc[1]+row_temp)
cv2.rectangle(model, max_loc, bottom_right_loc, (255, 0, 0), 5)
cv2.rectangle(res, max_loc, bottom_right_loc, 0, -1)
cv2.imshow("heat map", res)
cv2.waitKey(0)
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res)
bottom_right_loc2 = (max_loc2[0]+col_temp, max_loc2[1]+row_temp)
cv2.rectangle(model, max_loc2, bottom_right_loc2, (0, 255, 0), 5)
cv2.imshow("found!", model)
cv2.waitKey(0)"""