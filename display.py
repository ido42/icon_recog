import cv2
import os

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
        pass

    def show_display(self):
        cv2.imshow("display", self.image)
        cv2.waitKey(0)


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