import cv2 as cv
import numpy as np
import random

class DetectShape:
    def __init__(self, image_name):
        self.image = cv.imread(image_name)
        self.original = self.image.copy()
        self.output = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)

        self.w = self.image.shape[0]
        self.h = self.image.shape[1]
    
    def show_transformed_image(self):
        cv.imshow("Image", self.image)
        cv.waitKey()
    
    def show_output(self):
        cv.imshow("Image", self.output)
        cv.waitKey()
    
    def grey_image(self):
        self.image = cv.addWeighted(self.image, 2.5, self.image, 0, 0.2)
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
    
    def blur_image(self):
        # self.image = cv.GaussianBlur(self.image, (5, 5), 0)
        self.image = cv.bilateralFilter(self.image, 15, 75, 75)
    
    def edge_detection(self):
        v = np.median(self.image)
        sigma = 0.8

        threshold_one = int(max(0, (1.0 - sigma) * v))
        threshold_two = int(min(255, (1.0 - sigma) * v))

        self.image = cv.Canny(self.image, threshold_one, threshold_two)

    def clean_up(self, dilation_iterations=2, erode_iterations=1):
        kernel = np.ones((4, 5), dtype=np.uint8)
        d_im = cv.dilate(self.image, kernel, iterations=dilation_iterations)
        self.image = cv.erode(d_im, kernel, iterations=erode_iterations)
    
    def draw_lines(self):
        hough_lines = cv.HoughLines(self.image, 1, np.pi / 180, 70, None, 0, 0)

        for line in hough_lines:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                
                cv.line(self.image,(x1,y1),(x2,y2),(100,100,100), 2) 

    
    def detect_shapes(self):
        contours, _ = cv.findContours(self.image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w ,h) = cv.boundingRect(contour)
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))

            epsilon = 0.04 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)

            cv.drawContours(self.output, [approx], 0, (0, 255, 0), 2)
            cv.rectangle(self.original, (x, y), (x+w, y+h), color, 2)

            text = "Shape Not Recognized"
            if len(approx) == 3:
                text = "Triangle"
            elif len(approx) == 5:
                text = "Pentagon"

            font                    = cv.FONT_HERSHEY_SIMPLEX
            fontScale               = 1
            fontColor               = (0, 255, 0)
            thickness               = 1
            lineType                = 2
            text_width, text_height = cv.getTextSize(text, font, fontScale, lineType)[0]
            bottomLeftCornerOfText  = (min(x, self.h - text_width), y + h//2)

            cv.putText(self.original,text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
    
    def show_result(self):
        cv.imshow("Image", self.original)
        cv.waitKey()

if __name__ == '__main__':
    # shape_detector = DetectShape('Images/Blurry.jpg')
    # shape_detector = DetectShape('Images/TestCaseTrial.jpg')
    shape_detector = DetectShape('Images/Pentagon Small.jpg')
    shape_detector.show_transformed_image()
    shape_detector.grey_image()
    shape_detector.blur_image()
    shape_detector.blur_image()
    shape_detector.blur_image()
    shape_detector.edge_detection()
    shape_detector.clean_up(3, 1)
    shape_detector.detect_shapes()
    shape_detector.show_result()
    shape_detector.show_transformed_image()
