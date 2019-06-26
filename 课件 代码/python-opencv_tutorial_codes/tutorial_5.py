import cv2 as cv
import numpy as np


def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(copyImg, mask, (30, 30), (0, 255, 255), (50, 50, 50), (100, 100, 100),cv.FLOODFILL_FIXED_RANGE)
    # floodFill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None, flags=None)
    # src(seed.x,seed.y)-loDiff<=src(x,y)<=src(seed.x,seed.y)+upDiff
    cv.imshow("fill_color_demo", copyImg)


def fill_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300, :] = 255
    cv.imshow("fill_binary", image)

    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0
    cv.floodFill(image, mask, (200, 200), (100, 2, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("filled binary", image)

print("--------- Hello Python ---------")
src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# fill_color_demo(src)
fill_binary()
"""
face = src[150:400, 200:400]
gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # BGR转灰度
backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # 灰度转BGR
src[150:400, 200:400] = backface
cv.imshow("face", src)
"""
cv.waitKey(0)
cv.destroyAllWindows()
