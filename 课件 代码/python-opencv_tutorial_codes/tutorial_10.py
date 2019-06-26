import cv2 as cv
import numpy as np


def equalHist_demo(image):
    """直方图均值化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 基于灰度图，图像增强对比度的手段
    dst = cv.equalizeHist(gray)
    cv.imshow("equalHist_demo", dst)


def clahe_demo(image):
    """局部直方图均衡化"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # clipLimit对比度
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow("clahe_demo", dst)


def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    return rgbHist


def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    # 巴氏距离越小，约相似
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    # 相关性越接近1约相似
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    # 卡方越小越相似
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离: %s, 相关性: %s, 卡方: %s"%(match1, match2, match3))


print("--------- Hello Python ---------")
src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
equalHist_demo(src)
clahe_demo(src)

image1 = cv.imread("images/lena.png")
image2 = cv.imread("images/lena.png")
cv.imshow("image1", image1)
cv.imshow("image2", image2)
hist_compare(image1, image2)

cv.waitKey(0)

cv.destroyAllWindows()
