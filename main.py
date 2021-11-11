import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
def display(match):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(match)
    plt.show()


model_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1_display.PNG")).replace('\\', '/')
snowflake_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "snow_flake.PNG")).replace('\\', '/')
seven_segment_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "seven_segment.PNG")).replace('\\', '/')
model = cv2.imread(model_img_dir)
snowflake = cv2.imread(snowflake_img_dir)

model_cp = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
snowflake_cp = cv2.cvtColor(snowflake, cv2.COLOR_BGR2GRAY)

# straight forward template matching
rows, cols, channels = model.shape


model_cp = cv2.GaussianBlur(model_cp, (5, 5), cv2.BORDER_DEFAULT)
snowflake_cp = cv2.GaussianBlur(snowflake_cp, (5, 5), cv2.BORDER_DEFAULT)
snow_canny=cv2.Canny(snowflake_cp, 100, 200)
model_canny=cv2.Canny(model_cp, 100, 200)
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
cv2.waitKey(0)

# feature matching, brute force with orb descriptors
seven_segment = cv2.imread(seven_segment_dir)
seven_segment_cp=seven_segment.copy()

orb = cv2.ORB_create()
key_points1, descriptors1 = orb.detectAndCompute(seven_segment_cp, None) # detecying key points and descriptors for model and template
key_points2, descriptors2 = orb.detectAndCompute(model_cp, None)
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = brute_force.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance) # distance indicates similarity, here we sort them from most similar to least
ss_match = cv2.drawMatches(seven_segment_cp, key_points1, model_cp, key_points2, matches[:10], None, flags=2)
display(ss_match)

# sift matching
sift = cv2.SIFT_create() # needs extra opencv-contrib files
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

kp1_sift, des1_sift = sift.detectAndCompute(snowflake_cp, None)
kp2_sift, des2_sift = sift.detectAndCompute(model_cp, None)
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

img3 = cv2.drawMatchesKnn(snowflake_cp,kp1_sift,model_cp,kp2_sift,matches,None,**draw_params)

plt.imshow(img3,),plt.show()


# cv2.imshow("found", sf_match)
# cv2.waitkey(0)
"""f1 = np.fft.fft2(model)
f2 = np.fft.fft2(snowflake, (rows, cols))

lp = np.ones((5,5))

f_lp = np.fft.fft2(lp, (rows, cols))

cc = np.real(np.fft.ifft2(f_lp*f1*np.conj(f2)))
norm = np.zeros((100,100))
final = cv2.normalize(cc,  norm, 0, 255, cv2.NORM_MINMAX)

max_cc = cc.max()
[yshift, xshift] = np.nonzero(cc == max_cc)
print(yshift)
print(xshift)

cv2.circle(model, (int(yshift), int(xshift)), 30, 0,5)
cv2.imshow("model", model)

cv2.waitKey(0)

cv2.imshow("correlation", final)
cv2.waitKey(0)"""
