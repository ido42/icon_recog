import cv2
import numpy as np

model = cv2.imread("C:/Users/idil/PycharmProjects/icon_recog/images/f1_display.png")
snowflake=cv2.imread("C:/Users/idil/PycharmProjects/icon_recog/images/snow_flake.png")
model= cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
snowflake= cv2.cvtColor(snowflake, cv2.COLOR_BGR2GRAY)

f1 = np.fft.fft2(model)
f2 = np.fft.fft2(snowflake, (model.rows, model.cols))
lp= cv2.ones(5)
f_lp=np.fft.fft2(lp, (model.rows, model.cols))

cc = np.real(np.fft.ifft2(f_lp*f1*np.conj(f2)))

cv2.imshow(model)
cv2.waitKey(0)