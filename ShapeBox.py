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
        """Multiplies values in image and converts it to greyscale

        AddWeighted
            - addWeighted is used to perform a weighted sum of two images, but in this case, we're just using it
            - to multiply certain values in the image to make them more prominent once it's greyscaled.
        cvtColor
            - This is just greyscaling the image.
        """

        self.image = cv.addWeighted(self.image, 2.5, self.image, 0, 0.2)
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
    
    def blur_image(self):
        """Applies a blur

        Used to be implemented using a gaussian blur, but I've found that for detecting shapes in a difficult
        environment, a bilateral filter is better suited because it's more catered towards perserving edges
        and reducing any noise in the image. More sophisticated algorithms exist to do the same thing, but
        they're far slower.
        """

        # self.image = cv.GaussianBlur(self.image, (5, 5), 0)
        self.image = cv.bilateralFilter(self.image, 15, 75, 75)
    
    def edge_detection(self):
        """Edge Detection

        While the bilateral filter, in a lot of cases, manages to detect most edges out of the box, it's good
        and not all that time consuming to still apply edge detection to make edges even more prominent.

        *Note* In the implementation of cv.Canny, a gaussian filter is used to reduce noise.

        This method works as follows:
            1. Find median of all pixel values
            2. Set a sigma value (Explained later)
            3. Define two thresholds. In the implementation of cv.Canny, thresholds work as follows
                - Values above threshold_two are edges
                - Values below threshold_one are not edges
                - Values in between the two are only edges if connected to an already establish edge.
                - Sigma is just used to adjust the threshold. I played around with it for a bit and
                  found that around 0.8 is fine. There's no real reason for it, it just works well.
            4. Finally, cv.Canny takes care of edge detection using these parameters.
        """

        v = np.median(self.image)
        sigma = 0.8

        threshold_one = int(max(0, (1.0 - sigma) * v))
        threshold_two = int(min(255, (1.0 - sigma) * v))

        self.image = cv.Canny(self.image, threshold_one, threshold_two)

    def clean_up(self, dilation_iterations=2, erode_iterations=1):
        """Cleans up image by applying erosion and dilation

        This should be used exclusively after applying edge detection, or else it doesn't do a lot.

        DILATE
        --------
        Dilate, after edge detection, will make all lines bigger. I do this to remove and gaps in shapes that we
        detect.

        ERODE
        --------
        Erosion, after edge detection, will make lines thinner. This is applied after dilation to reduce the size
        of the lines again to normal. After completing these operations in succession, lines that should be connected
        will have a higher chance of actually being connected. 

        *Note* (This can also introduce fake shapes as well)

        KERNEL
        --------
        This determines the size of the matrix that scans over the image and applies the erode and dilate filters.
        """

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
