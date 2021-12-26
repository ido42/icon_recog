import cv2
import os
import numpy as np
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

class display():

    def __init__(self, template, buttons_loc, screen_loc):
        self.temp = cv2.imread(template)  # load empty template image without buttons sand icons
        self.button_list = buttons_loc  # locations of the buttons in an list ([x1, y1, x2, y2], ....)
        self.screen = screen_loc  # the location of the screen [x1, y1, x2, y2]
        self.image = cv2.imread(template)
        cv2.rectangle(self.image, (self.screen[0], self.screen[1]), (self.screen[2], self.screen[3]), (255, 0, 0), 1)

        self.icons = None

    def set_buttons(self, butt_images):  # put images of the buttons to the relative locations
        for i in range(len(butt_images)):
            b = cv2.imread(butt_images[i])
            dim = (self.button_list[i][3]-self.button_list[i][1], self.button_list[i][2]-self.button_list[i][0])
            b = cv2.resize(b, dim, interpolation=cv2.INTER_AREA)
            self.image[self.button_list[i][0]:self.button_list[i][0]+b.shape[0], self.button_list[i][1]:self.button_list[i][1]+b.shape[1]] = b

    def set_icons(self, icons):
        self.icons = icons

    def find_on_camera(self,camera_view):
        camera_view=cv2.cvtColor(camera_view, cv2.COLOR_BGR2GRAY)
        scale_percent = 0.36  # percent of original size
        width = int(camera_view.shape[1] * scale_percent)
        height = int(camera_view.shape[0] * scale_percent)
        dim = (width, height)
        model = cv2.resize(camera_view, dim, interpolation=cv2.INTER_AREA)

        img=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        M = cv2.getOptimalDFTSize(camera_view.shape[0])
        N = cv2.getOptimalDFTSize(camera_view.shape[1])
        ft_model = forward_fft(camera_view, M, N)
        ft_sf = forward_fft(img, M, N)

        mag1, p1 = cv2.cartToPolar(np.real(ft_model), np.imag(ft_model))
        mag2, p2 = cv2.cartToPolar(np.real(ft_sf), np.imag(ft_sf))

        hp = highpass(mag1.shape[0], mag1.shape[1])
        h_mag1 = mag1 * hp
        h_mag2 = mag2 * hp
        norm = np.zeros((800, 800))
        h_mag1 = cv2.normalize(h_mag1, norm, 0, 255, cv2.NORM_MINMAX)
        h_mag2 = cv2.normalize(h_mag2, norm, 0, 255, cv2.NORM_MINMAX)

        log_f1, l1 = logpol(h_mag1)
        log_f2, l2 = logpol(h_mag2)

        rotation_and_scale = cv2.phaseCorrelate(log_f2, log_f1)
        ang=rotation_and_scale[0][1] / log_f1.shape[1] * 180
        scale=pow(l2, rotation_and_scale[0][0] / log_f1.shape[0])
        scale_percent = 0.36  # percent of original size
        width = int(img.shape[1] / scale_percent)
        height = int(img.shape[0] / scale_percent)
        dim = (width, height)
        #camera_view = cv2.resize(camera_view, dim,interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(camera_view, img, cv2.TM_CCOEFF)
        cv2.imshow("heat map", res)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        row_temp, col_temp = img.shape
        bottom_right_loc = (max_loc[0] + col_temp, max_loc[1] + row_temp)
        cv2.rectangle(camera_view, max_loc, bottom_right_loc, 255, 5)
        return ang,scale

    def show_display(self):
        cv2.imshow("display", self.image)
        cv2.waitKey(0)

real_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "ex3.jpeg")).replace('\\', '/')
model_img_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "f1template.png")).replace('\\', '/')
snowflake_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "buttons", "snow_flake.png")).replace('\\', '/')
eco_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "buttons", "timer.png")).replace('\\', '/')
wifi_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "icon_recog", "images", "buttons", "wifi.png")).replace('\\', '/')
b_dir = [snowflake_dir, eco_dir, wifi_dir]
b_loc = ([164, 90, 277, 153], [164, 225, 277, 288], [164, 293, 277, 293+63])
screen_loc = [0, 0, 580, 163]
d = display(model_img_dir, b_loc,screen_loc)
d.set_buttons(b_dir)
d.show_display()
cam=cv2.imread(real_img_dir)
d.find_on_camera(cam)
cv2.imshow("",cam)
cv2.waitKey(0)