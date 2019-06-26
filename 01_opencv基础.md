# 1. 图片与视频的读取

## 1.1图片读取

```python
import cv2 as cv
# 读取图片，src是[512，512，3]的数组
src = cv.imread("images/lena.png")
# 窗口大小会自动调整以适合被显示图像，0，可以手动调节，1，不可以手动调节
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE) 
# 显示图片
cv.imshow("input image", src)
cv.waitKey(0) # 不运行其他代码;直到键盘值为key的响应之后。
cv.destroyAllWindows() # 关闭所有窗口

```

## 1.2 视频读取

```python
# 打开笔记本相机
capture = cv.VideoCapture(0)
while(True):
    ret, frame = capture.read()   
    # 左右调换
    frame = cv.flip(frame, 1) 
    cv.imshow("video", frame)
    c = cv.waitKey(50)
    print("c=",c)
    if c == 27:
     	 break
```

# 2. 时间

```python
t1 = cv.getTickCount()
...
t2 = cv.getTickCount()
time = (t2 - t1) / cv.getTickFrequency()
print("time : %s ms" % (time * 1000))
```



# 3. 色彩转换

## 3.1 色彩转换

```Python
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow("hsv", hsv)
yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
cv.imshow("yuv", yuv)
Ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
cv.imshow("ycrcb", Ycrcb)
```

## 3.2 图片像素取反

```python
dst = cv.bitwise_not(image)
cv.imshow("inverse demo", dst)
```

## 3.3 色彩分离与合并

```python
b, g, r = cv.split(src)
cv.imshow("blue", b)
cv.imshow("green", g)
cv.imshow("red", r)

src = cv.merge([b, g, r])
src[:, :, 0] = 0
cv.imshow("changed image", src)
```

## 3.4 简单的对象抽取

```python
def extrace_object_demo():
    capture = cv.VideoCapture("D:/vcprojects/images/video_006.mp4")
    while True:
        ret, frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 把绿色给过滤出来
        lower_hsv = np.array([37, 43, 46])
        upper_hsv = np.array([77, 255, 255])
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        dst = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("video", frame)
        cv.imshow("mask", dst)
        c = cv.waitKey(40)
        if c == 27:
            break
```

## 3.5 像素运算

### 3.5.1 加减乘除

```python

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
    
src1 = cv.imread("images/LinuxLogo.jpg")
src2 = cv.imread("images/WindowsLogo.jpg")
# print(src1.shape)
# print(src2.shape)
cv.namedWindow("image1", cv.WINDOW_AUTOSIZE)
cv.imshow("image1", src1)
cv.imshow("image2", src2)
add_demo(src1, src2)
subtract_demo(src1, src2)
divide_demo(src1, src2)
multiply_demo(src1, src2)
```

### 3.5.2 对比度

```python
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
src = cv.imread("images/lena.png")
cv.imshow("image2", src)
contrast_brightness_demo(src, 15, 100)
```

# 4.ROI 和泛洪填充

## 4.1 ROI(Region of Interset)

```python
src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

face = src[150:400, 200:400]
gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # BGR转灰度
backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # 灰度转BGR
src[150:400, 200:400] = backface
cv.imshow("face", src) 
```

## 4.2 泛洪填充

```python

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
    
fill_color_demo(src)
fill_binary()
```

# 5 模糊

## 5.1模糊操作

```python

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
		# 自定义卷积核
    #kernel = np.ones([5, 5], np.float32)/25
    # 锐化算子
    kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]], np.float32)
    dst = cv.filter2D(image, -1, kernel=kernel) # ddepth=-1:深度与源图相同

    cv.imshow("custom_blur_demo", dst)

src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
blur_demo(src)
median_blur_demo(src)
custom_blur_demo(src)
```

## 5.2高斯模糊

```python 
# 添加高斯噪声
def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv


def gaussian_noise(image):
    """加高斯噪声"""
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv.imshow("noise image", image)

src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

t1 = cv.getTickCount()
gaussian_noise(src) # 添加高斯噪声
t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
print("time consume : %s"%(time*1000))
# dst = cv.GaussianBlur(src, (0, 0), 15)
dst = cv.GaussianBlur(src, (3, 3), 0)
# cv2.GaussianBlur(src, ksize, sigmaX) ksize和sigmaX可以互相算，给了一个另外就给0
cv.imshow("Gaussian Blur", dst)
```

## 5.3 边缘保留滤波(EPF)

```python

def bi_demo(image):
    # 高斯双边
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
    
src = cv.imread("images/lqq.jpeg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# bi_demo(src)
shift_demo(src)
```

# 6.图像直方图

抓住图像的特征像素分布

```python

def plot_demo(image):
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


def image_hist(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        print(i,color)
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
src = cv.imread("images/cxy.jpeg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
plot_demo(src)
# print()
image_hist(src)
```

## 6.1直方图应用

### 6.1.1调整对比度：直方图均衡化

```python

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
    
src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
equalHist_demo(src)
clahe_demo(src)
```

### 6.1.2 直方图比较

```python

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

image1 = cv.imread("images/lena.png")
image2 = cv.imread("images/lena.png")
cv.imshow("image1", image1)
cv.imshow("image2", image2)
hist_compare(image1, image2)
```

## 6.2直方图反向投影

需要在HSV色彩空间

```python 


def back_projection_demo():
    sample = cv.imread("D:/javaopencv/sample.png")
    target = cv.imread("D:/javaopencv/target.png")
    # 转到hsv色彩空间
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    # show images
    cv.imshow("sample", sample)
    cv.imshow("target", target)

    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    # 归一化
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv.imshow("backProjectionDemo", dst)

```

# 7.模板匹配

```python

def template_demo():
    tpl = cv.imread("D:/javaopencv/tpl.png")
    target = cv.imread("D:/javaopencv/test1.png")
    cv.imshow("template image", tpl)
    cv.imshow("target image", target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        # 计算最佳位置
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0]+tw, tl[1]+th);
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        cv.imshow("match-"+np.str(md), target)
        # cv.imshow("match-" + np.str(md), result)

```

# 8.二值化

## 8.1 图像二值化

二值图像阈值可以参考直方图；

图像二值化的方法：

- OTUS
- Triangle
- 自动和手动

```python

def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    # cv.THRESH_OTSU起作用，127就不起作用了
    print("threshold value %s"%ret)
    cv.imshow("binary", binary)


def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 		  cv.THRESH_BINARY, 25, 10)
    cv.imshow("binary", binary)


def custom_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w*h])
    mean = m.sum() / (w*h)
    print("mean : ", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("binary", binary)
    
    
src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

threshold_demo(src)
local_threshold(src)
custom_threshold(src)
```

## 8.2超大图像二值化

```python


def big_image_binary(image):
    print(image.shape)
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:cw+col]
            print(np.std(roi), np.mean(roi))
            dst = cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,127,20)
            """空白图像过滤
            dev = np.std(roi)
            if dev < 15:
                gray[row:row + ch, col:cw + col] = 255
            else:
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                gray[row:row + ch, col:cw + col] = dst
            """
            gray[row:row + ch, col:cw + col] = dst
    cv.imwrite("D:/vcprojects/result_binary.png", gray)

src = cv.imread("images/red_text2.png")
big_image_binary(src)
```

# 9.图像金字塔

## 9.1 高斯金字塔

```python

def pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_"+str(i), dst)
        temp = dst.copy()
    return pyramid_images

```



## 9.2 拉普拉斯金字塔

```pytHon
def lapalian_demo(image):
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0 :
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("lapalian_down_" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("lapalian_down_"+str(i), lpls)
            
src = cv.imread("images/lena.png") # 图片必须是512*512的
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
lapalian_demo(src)
```

# 10.图像梯度

## 10.1一阶导数和Sobel算子

```python

def sobel_demo(image):
    # cv.Scharr是Sobel算子的增强版本
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient-x", gradx)
    cv.imshow("gradient-y", grady)
    # 梯度加权
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradient", gradxy)
```



## 10.2 二阶导数与拉普拉斯算子

```python

def lapalian_demo(image):
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    # 自己定义拉普拉斯算子
    # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    # lpls = cv.convertScaleAbs(dst)
    cv.imshow("lapalian_demo", lpls)
```



# 11.边缘提取

## 11.1Canny算法

### 步骤：

1. 高斯模糊
2. 灰度转换
3. 计算梯度
4. 非最大信号抑制
5. 高低阈值输出二值图像

```python
def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # X Gradient
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # Y Gradient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    #edge
    # edge_output = cv.Canny(xgrad, ygrad, 50, 150) # 可以用梯度，也可以传blurred,gray
    edge_output = cv.Canny(gray, 50, 150)
    cv.imshow("Canny Edge", edge_output)

    dst = cv.bitwise_and(image, image, mask=edge_output)
    cv.imshow("Color Edge", dst)
```

## 11.2直线检测

### 霍夫直线变换

```python

def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    for line in lines:
        print(type(lines))
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("image-lines", image)


def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        print(type(line))
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible_demo", image)

```

## 11.3圆检测

```python

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
```

## 11.4 轮廓发现

```python


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # X Gradient
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # Y Gradient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    #edge
    #edge_output = cv.Canny(xgrad, ygrad, 50, 150)
    edge_output = cv.Canny(gray, 30, 100)
    cv.imshow("Canny Edge", edge_output)
    return edge_output


def contours_demo(image):
    """dst = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary image", binary)"""
    binary = edge_demo(image)
    print(binary.shape)
		# 格式错误的时候需要怎么处理 binary = np.array(binary, np.uint8)
    cloneImage, contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)
        approxCurve = cv.approxPolyDP(contour, 4, True)
        if approxCurve.shape[0] > 6:
            cv.drawContours(image, contours, i, (0, 255, 255), 2)
        if approxCurve.shape[0] == 4:
            cv.drawContours(image, contours, i, (255, 255, 0), 2)
        print(approxCurve.shape[0])
        print(i)
    cv.imshow("detect contours", image)
```

# 12.对象测量

```python

def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value : %s"%ret)
    cv.imshow("binary image", binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    outImage, contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        rate = min(w, h)/max(w, h)
        print("rectangle rate : %s"%rate)
        mm = cv.moments(contour)
        print(type(mm))
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']
        cv.circle(dst, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)
        #cv.rectangle(dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print("contour area %s"%area)
        approxCurve = cv.approxPolyDP(contour,4, True)
        print(approxCurve.shape)
        if approxCurve.shape[0] > 6:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        if approxCurve.shape[0] == 4:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)
        if approxCurve.shape[0] == 3:
            cv.drawContours(dst, contours, i, (255, 0, 0), 2)
    cv.imshow("measure-contours", dst)


```

# 13.图像形态学

## 13.1 膨胀

3x3的结构元素模板，支持任意形状的结构元素

### 膨胀的作用

- 对象大小增加一个像素(3x3)

- 平滑对象边缘

-  减少或者填充对象之间的距离

```python

def dilate_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    # 得到结构元素,一般选择MORPH_RECT
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernel)
    cv.imshow("dilate_demo", dst)
```



## 13.2 腐蚀

### 腐蚀的作用

- 对象大小减少1个像素(3x3)

- 平滑对象边缘

- 弱化或者分割图像之间的半岛型连接

```python

def erode_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    # 得到结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.erode(binary, kernel)
    cv.imshow("erode_demo", dst)

```

### 注意 膨胀和腐蚀都可以在彩色图上操作

```python
src = cv.imread("images/lena.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
# erode_demo(src)
# dilate_demo(src)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dst = cv.erode(src, kernel)
```

## 13.3 开闭操作

开操作=腐蚀+膨胀 —>去除小的干扰块儿

闭操作=膨胀+腐蚀 —>填充闭合区域

```python

def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))  # (15,1)提取水平线,(1,15)提取竖直线
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))  # cv.MORPH_ELLIPSE 保留圆
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("open-result", binary)


def close_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("close_demo", binary)

```

 ## 13.4 顶帽(tophat)与黑帽(blackhat)

顶帽：原图像与开操作之间的 差值图像

黑帽：原图像与闭操作之间的插值图像

```python
def top_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 120
    dst = cv.add(dst, cimage)
    cv.imshow("tophat", dst)
def black_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 120
    dst = cv.add(dst, cimage)
    cv.imshow("blackhat", dst)


```



## 13.5形态学梯度

### 基本梯度

膨胀后的图像减去腐蚀后的图像得到的差值图像

### 内部梯度

原图减去腐蚀后的图像得到的差值图像

### 外部梯度

图像膨胀后的图像减去原图得到的差值图像

```python 

def gradient2_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)
    dst1 = cv.subtract(image, em) # internal gradient
    dst2 = cv.subtract(dm, image) # external gradient
    cv.imshow("internal", dst1)
    cv.imshow("external", dst2)

```

# 14.分水岭算法

### 步骤

输入图像—>灰度—>二值—>距离变换—>寻找种子—>生成marker—>分水岭变换—>输出图像

```python

def watershed_demo():
    # remove noise if any
    print(src.shape)
    blurred = cv.pyrMeanShiftFiltering(src, 10, 100)
    # gray\binary image
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary-image", binary)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(mb, kernel, iterations=3)
    cv.imshow("mor-opt", sure_bg)

    # distance transform
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("distance-t", dist_output * 50)

    ret, surface = cv.threshold(dist, dist.max() * 0.6, 255, cv.THRESH_BINARY)

    surface_fg = np.uint8(surface)
    cv.imshow("surface-bin", surface_fg)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)

    # watershed transform
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers == -1] = [0, 0, 255]
    cv.imshow("result", src)

```

# 15.人脸检测

### 图像人脸检测

```python
def face_detect_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("./images/haarcascade_frontalface_alt_tree.xml")
    faces = face_detector.detectMultiScale(gray, 1.1, 2)
    for x, y, w, h in faces:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv.imshow("result", image)
```

### 视频人脸检测

```python
capture = cv.VideoCapture(0)
while(True):
    ret, frame = capture.read()
    frame = cv.flip(frame, 1)
    face_detect_demo(frame)
    c = cv.waitKey(10)
    if c == 27: # ESC
        break
```

