import cv2 as cv
import numpy as np


def detect_circles_demo(image):
    # 中值滤波
    dst = cv.pyrMeanShiftFiltering(image, 200, 400)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow("circles", image)


print("--------- Python OpenCV Tutorial ---------")
src = cv.imread("images/coin_001.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
detect_circles_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()
