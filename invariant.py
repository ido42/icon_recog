import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import cmath
model_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1_display.PNG")).replace('\\', '/')
snowflake_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "snow_flake.PNG")).replace('\\', '/')
seven_segment_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "seven_segment.PNG")).replace('\\', '/')
model = cv2.imread(model_img_dir)

snowflake = cv2.imread(snowflake_img_dir)
model_cp = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
snowflake_cp = cv2.cvtColor(snowflake, cv2.COLOR_BGR2GRAY)

scale_percent = 1.5 # percent of original size

width = int(snowflake_cp.shape[1] * scale_percent)
height = int(snowflake_cp.shape[0] * scale_percent)
dim = (width, height)
M = cv2.getRotationMatrix2D((width/2, height/2), 20, 1.0)
rotated = cv2.warpAffine(snowflake_cp, M, (width, height))
cv2.imshow("Rotated by 45 Degrees", rotated)
snowflake_cp = cv2.resize(snowflake_cp, dim, interpolation=cv2.INTER_AREA)

#sf_shift=np.fft.fftshift(snowflake_cp)
#model_shift=np.fft.fftshift(model_cp)
rows, cols, channels = model.shape
f1 = np.fft.fft2(model_cp,(cols,cols))
f2 = np.fft.fft2(snowflake_cp, (cols, cols))
f2=np.fft.fftshift(f2)
f1=np.fft.fftshift(f1)
#High-pass Gaussian filter
(P, Q) = f1.shape
H = np.zeros((P,Q))
D0 = 40
for u in range(P):
    for v in range(Q):
        H[u, v] = 1.0 - np.exp(- ((u - P / 2.0) ** 2 + (v - Q / 2.0) ** 2) / (2 * (D0 ** 2)))
k1 = 0.5 ; k2 = 0.75
HFEfilt = k1 + k2 * H # Apply High-frequency emphasis

f1_hp=HFEfilt*f1[0]
f2_hp=HFEfilt*f2[0]
cv2.imshow("hp f", np.real(f1_hp))
cv2.imshow("hp f2", np.real(f2_hp))
cv2.imshow("model f", np.abs(f1))
cv2.imshow("snow f", np.abs(f2))

cv2.waitKey(0)

value1 = np.sqrt(((f1.shape[0]/2.0)**2.0)+((f1.shape[1]/2.0)**2.0))
value2 = np.sqrt(((f2.shape[0]/2.0)**2.0)+((f2.shape[1]/2.0)**2.0))

log_f1_r = cv2.warpPolar(np.real(f1_hp),(f1.shape[0], f1.shape[1]), (f1.shape[0]/2, f1.shape[1]/2), 800,cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
log_f2_r = cv2.warpPolar(np.real(f2_hp),(f2.shape[0], f2.shape[1]), (f2.shape[0]/2, f2.shape[1]/2), 800,cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
log_f1_i = cv2.warpPolar(np.imag(f1_hp),(f1.shape[0], f1.shape[1]), (f1.shape[0]/2, f1.shape[1]/2), 800,cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)
log_f2_i = cv2.warpPolar(np.imag(f2_hp),(f2.shape[0], f2.shape[1]), (f2.shape[0]/2, f2.shape[1]/2), 800,cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS+ cv2.WARP_POLAR_LOG)

log_f1=log_f1_r+1j*log_f1_i
log_f2=log_f2_r+1j*log_f2_i

res=log_f1_r*np.conj(log_f2_r)
res_inv=np.fft.ifft2(res)
cv2.imshow("snow edge", log_f1_r)
cv2.imshow("real edge", log_f2_r)
cv2.waitKey(0)
#[a,e]=cv2.phaseCorrelate(log_f1_r,log_f2_r)
#print(a,",",e)
#res = cv2.matchTemplate(log_f1_r.astype(np.uint8), log_f2_r.astype(np.uint8), cv2.TM_CCOEFF)
#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
cv2.imshow("res", res)

cv2.imshow("res i", np.real(res_inv))

cv2.waitKey(0)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(np.real(res_inv))
print((max_loc[1])/res_inv.shape[1]*360,",",np.exp(max_loc[0]/res_inv.shape[0]))
"""
print(np.exp(max_loc[1]/log_f1.shape[1]))"""


