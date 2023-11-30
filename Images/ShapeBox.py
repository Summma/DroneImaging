import cv2 as cv
import numpy as np

class DetectShape:
    def __init__(self, image_name):
        self.image = cv.imread(image_name)
    
    def show_image(self):
        cv.imshow("Image", self.image)
        cv.waitKey()
    
    def grey_image(self):
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
    
    def blur_image(self):
        self.image = cv.GaussianBlur(self.image, (5,5), 0)
    
    def edge_detection(self, threshold_one, threshold_two):
        self.image = cv.Canny(self.image, threshold_one, threshold_two)
    
    def detect_shapes(self):
        contours, _ = cv.findContours(self.image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
