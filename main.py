import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

model_img_dir= os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1_display.PNG")).replace('\\', '/')
snowflake_img_dir=os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "snow_flake.PNG")).replace('\\', '/')

model = cv2.imread("C:/Users/idil/PycharmProjects/icon_recog/images/f1_display.PNG")
snowflake = cv2.imread("C:/Users/idil/PycharmProjects/icon_recog/images/snow_flake.png")
model_cp = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
snowflake_cp = cv2.cvtColor(snowflake, cv2.COLOR_BGR2GRAY)

# straight forward template matching
rows, cols,channels = model.shape

# the methods that can be used are cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_SQDIFF etc.
res = cv2.matchTemplate(model_cp, snowflake_cp, cv2.TM_CCOEFF)
cv2.imshow("heat map", res)
cv2.waitKey(0)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
row_temp, col_temp = snowflake_cp.shape
bottom_right_loc = (max_loc[0]+col_temp, max_loc[1]+row_temp)
cv2.rectangle(model,max_loc,bottom_right_loc,(255,0,0),5)
cv2.imshow("found!", model)
cv2.waitKey(0)

#feature matching, brute force with orb descriptors
orb = cv2.ORB_create()
key_points1, descriptors1 = orb.detectAndCompute(snowflake_cp, None) # detecying key points and descriptors for model and template
key_points2, descriptors2 = orb.detectAndCompute(model_cp, None)
brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = brute_force.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance) # distance indicates similarity, here we sort them from most similar to least
sf_match=cv2.drawMatches(snowflake_cp, key_points1,model_cp, key_points2, matches[:10],None,flags=2)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
plt.imshow(sf_match)
plt.show()
#cv2.imshow("found", sf_match)
#cv2.waitkey(0)
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
