import cv2 as cv
import numpy as np


def add_demo(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)


def subtract_demo(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)


def divide_demo(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("divide_demo", dst)


def multiply_demo(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("multiply_demo", dst)


def logic_demo(m1, m2):
    # 与
    dst = cv.bitwise_and(m1, m2)
    # # 或
    # # dst = cv.bitwise_or(m1, m2)
    # image = cv.imread("images/lena.png")
    # cv.imshow("image1", image)
    # # 非
    # dst = cv.bitwise_not(image)
    cv.imshow("logic_demo", dst)


def contrast_brightness_demo(image, c, b):
    """
    提升亮度和对比度
    :param image:
    :param c: 对比度
    :param b: 亮度
    :return:
    """
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow("con-bri-demo", dst)


def others(m1, m2):
    M1, dev1 = cv.meanStdDev(m1)
    M2, dev2 = cv.meanStdDev(m2)
    h, w = m1.shape[:2]

    print(M1)
    print(M2)

    print(dev1)
    print(dev2)

    img = np.zeros([h, w], np.uint8)
    m, dev = cv.meanStdDev(img)
    print(m)
    print(dev)

print("--------- Hello Python ---------")
src1 = cv.imread("images/LinuxLogo.jpg")
src2 = cv.imread("images/WindowsLogo.jpg")
# print(src1.shape)
# print(src2.shape)
cv.namedWindow("image1", cv.WINDOW_AUTOSIZE)
cv.imshow("image1", src1)
cv.imshow("image2", src2)


# add_demo(src1, src2)
# subtract_demo(src1, src2)
# divide_demo(src1,src2)
# multiply_demo(src1,src2)
# others(src1, src2)
# logic_demo(src1, src2)

src = cv.imread("images/lena.png")
cv.imshow("image2", src)
contrast_brightness_demo(src, 15, 100)



cv.waitKey(0)

cv.destroyAllWindows()
