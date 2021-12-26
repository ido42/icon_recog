import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math
def forward_fft(image,M,N):
#    M = cv2.getOptimalDFTSize(image.shape[0])
#    N = cv2.getOptimalDFTSize(image.shape[1])
    padded_img = cv2.copyMakeBorder(image, 0, M-image.shape[0], 0, N-image.shape[1], cv2.BORDER_CONSTANT, None, value=0)
    cv2.imshow("",padded_img)
    cv2.waitKey(0)
    ft = np.fft.fft(padded_img)
    cv2.imshow("",np.abs(ft))
    cv2.waitKey(0)
    ft_shift = np.fft.fftshift(ft)
    cv2.imshow("", np.abs(ft_shift))
    cv2.waitKey(0)
    #ft_shift = ft_shift/(M*N)
    #cv2.imshow("", np.abs(ft_shift))
    #cv2.waitKey(0)
    return ft_shift

def highpass(r,c):
    a = np.zeros((r, 1))
    b = np.zeros((1, c))
    step_y = math.pi/r
    val = -math.pi*0.5
    for i in range(r):
        a[i] = math.cos(val)
        val += step_y

    step_x = math.pi / c
    val = -math.pi * 0.5
    for i in range(c):
        b[0][i] = math.cos(val)
        val += step_x
    tmp = np.matmul(a, b)
    hp = (1-tmp)*(2-tmp)
    return hp

def logpol(src):
    radii = src.shape[1]
    angles = src.shape[0]
    center = (round(radii / 2), round(angles/ 2))
    d = np.sqrt((radii - center[0])**2 + (angles - center[1])**2)
    log_base = pow(10.0, np.log10(d) / radii)
    d_theta = math.pi / angles
    theta = math.pi / 2.0
    radius = 0
    map_x = np.zeros(np.shape(src),np.float32)
    map_y = np.zeros(np.shape(src),np.float32)
    for i in range(angles):
        for j in range(radii):
            radius = pow(log_base, j)
            x = radius * math.sin(theta) + center[0]
            y = radius * math.cos(theta) + center[1]
            map_x[i][j] = x
            map_y[i][j] = y

        theta += d_theta


    dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
    return dst, log_base

model_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1_display_real_cropped.jpeg")).replace('\\', '/')
snowflake_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images","buttons", "snow_flake.png")).replace('\\', '/')
model = cv2.imread(model_img_dir)
scale_percent = 0.36   # percent of original size
width = int(model.shape[1] * scale_percent)
height = int(model.shape[0] * scale_percent)
dim = (width, height)
model = cv2.resize(model, dim, interpolation=cv2.INTER_AREA)

snowflake = cv2.imread(snowflake_img_dir)
model_cp = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
snowflake_cp = cv2.cvtColor(snowflake, cv2.COLOR_BGR2GRAY)

scale_percent = 1.2  # percent of original size

w = int(snowflake_cp.shape[1] * scale_percent)
h = int(snowflake_cp.shape[0] * scale_percent)
dim = (w, h)
M = cv2.getRotationMatrix2D((w/2, h/2), 180, 1.0)
snowflake_cp = cv2.warpAffine(snowflake_cp, M, (width, height))

snowflake_cp = np.float32(snowflake_cp)/255
model_cp = np.float32(model_cp)/255

M = cv2.getOptimalDFTSize(model_cp.shape[0])
N = cv2.getOptimalDFTSize(model_cp.shape[1])

ft_model = forward_fft(model_cp, M, N)
ft_sf = forward_fft(snowflake_cp, M, N)

mag1, p1 = cv2.cartToPolar(np.real(ft_model), np.imag(ft_model))
mag2, p2 = cv2.cartToPolar(np.real(ft_sf), np.imag(ft_sf))

hp = highpass(mag1.shape[0],mag1.shape[1])
h_mag1 = mag1*hp
h_mag2 = mag2*hp
norm = np.zeros((800, 800))
h_mag1 = cv2.normalize(h_mag1,  norm, 0, 255, cv2.NORM_MINMAX)
h_mag2 = cv2.normalize(h_mag2,  norm, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("1", h_mag1)
cv2.imshow("2", h_mag2)
cv2.waitKey(0)

value1 = np.sqrt(((h_mag1.shape[0]/2.0)**2.0)+((h_mag1.shape[1]/2.0)**2.0))
value2 = np.sqrt(((h_mag2.shape[0]/2.0)**2.0)+((h_mag2.shape[1]/2.0)**2.0))


log_f1 = cv2.warpPolar(h_mag1,(h_mag1.shape[0], h_mag1.shape[1]), (h_mag1.shape[0]/2, h_mag1.shape[1]/2), value1,cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
log_f2 = cv2.warpPolar(h_mag2,(h_mag2.shape[0], h_mag2.shape[1]), (h_mag2.shape[0]/2, h_mag2.shape[1]/2), value2,cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
log_f1, l1 = logpol(h_mag1)
log_f2, l2 = logpol(h_mag2)

rotation_and_scale = cv2.phaseCorrelate(log_f2, log_f1)
print(rotation_and_scale[0][1]/log_f1.shape[1]*180, "\n")
print(pow(l2,rotation_and_scale[0][0]/log_f1.shape[0]))

"""
# here i just test some of the functions, not really an important file
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
"""
"""
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
row_temp, col_temp = snowflake_cp.shape
bottom_right_loc = (max_loc[0]+col_temp, max_loc[1]+row_temp)
cv2.rectangle(model, max_loc, bottom_right_loc, (255, 0, 0), 5)

cv2.imshow("found!", model)
cv2.waitKey(0)
"""