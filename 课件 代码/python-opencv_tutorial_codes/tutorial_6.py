import cv2 as cv
import numpy as np


def blur_demo(image):
    # 实际上卷积（5，5）
    # 均值模糊，去随机噪声
    dst = cv.blur(image, (5, 5))
    cv.imshow("blur_demo", dst)


def median_blur_demo(image):
    # 中值模糊，去椒盐噪声
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur_demo", dst)


def custom_blur_demo(image):
    #kernel = np.ones([5, 5], np.float32)/25
    # 锐化算子
    kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]], np.float32)
    dst = cv.filter2D(image, -1, kernel=kernel) # ddepth=-1:深度与源图相同

    cv.imshow("custom_blur_demo", dst)


print("--------- Hello Python ---------")
src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
blur_demo(src)
median_blur_demo(src)
custom_blur_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()