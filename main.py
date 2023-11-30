import cv2 as cv
import numpy as np

from ShapeBox import DetectShape

if __name__ == '__main__':
    shape_detector = DetectShape('Images/Pentagon Small.jpg')
    shape_detector.grey_image()
    shape_detector.blur_image()
    shape_detector.edge_detection(0.01, 0.03)
    shape_detector.show_image()