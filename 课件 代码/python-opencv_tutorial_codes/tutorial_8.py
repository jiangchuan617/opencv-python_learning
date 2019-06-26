import cv2 as cv
import numpy as np


def bi_demo(image):
    # 边缘保留滤波EPF
    # bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None):
    dst = cv.bilateralFilter(image, 0, 100, 15)
    # d=0时候，由sigmaSpace推导
    # dst = dst[::2,::2,:]
    cv.imshow("bi_demo", dst)


def shift_demo(image):
    # pyrMeanShiftFiltering(src, sp, sr, dst=None, maxLevel=None, termcrit=None)
    # 均值迁移
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("shift_demo", dst)


print("--------- Hello Python ---------")
src = cv.imread("images/lqq.jpeg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# bi_demo(src)
shift_demo(src)
cv.waitKey(0)

cv.destroyAllWindows()
