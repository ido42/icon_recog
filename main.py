import cv2
import numpy as np

model = cv2.imread("C:/Users/DORA/PycharmProjects/icon_recog/images/f1_display.PNG")
snowflake = cv2.imread("C:/Users/DORA/PycharmProjects/icon_recog/images/snow_flake.png")
model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
snowflake = cv2.cvtColor(snowflake, cv2.COLOR_BGR2GRAY)

rows, cols = model.shape
f1 = np.fft.fft2(model)
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

cv2.circle(model, (xshift, yshift), 30, 0)
cv2.imshow("model", model)

cv2.waitKey(0)

cv2.imshow("correlation", final)
cv2.waitKey(0)
