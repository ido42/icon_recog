import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
model_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1_display_real_cropped.jpeg")).replace('\\', '/')
snowflake_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "snow_flake.jpg")).replace('\\', '/')
model = cv2.imread(model_img_dir)

scale_percent = 0.36 # percent of original size

width = int(model.shape[1] * scale_percent)
height = int(model.shape[0] * scale_percent)
dim = (width, height)
model = cv2.resize(model, dim, interpolation=cv2.INTER_AREA)
snowflake = cv2.imread(snowflake_img_dir)
model_cp = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
snowflake_cp = cv2.cvtColor(snowflake, cv2.COLOR_BGR2GRAY)

cv2.imshow("model", model)
cv2.imshow("sf", snowflake)
cv2.waitKey(0)

model_gs = cv2.GaussianBlur(model_cp, (5, 5), cv2.BORDER_DEFAULT)
snowflake_gs = cv2.GaussianBlur(snowflake_cp, (5, 5), cv2.BORDER_DEFAULT)
snow_canny=cv2.Canny(snowflake_gs, 100, 200)
model_canny=cv2.Canny(model_gs, 100, 200)
res = cv2.matchTemplate(model_canny, snow_canny, cv2.TM_CCOEFF)
cv2.imshow("heat map", res)
cv2.waitKey(0)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
row_temp, col_temp = snowflake_cp.shape
bottom_right_loc = (max_loc[0]+col_temp, max_loc[1]+row_temp)
cv2.rectangle(model, max_loc, bottom_right_loc, (255, 0, 0), 5)

cv2.imshow("found!", model)
cv2.waitKey(0)