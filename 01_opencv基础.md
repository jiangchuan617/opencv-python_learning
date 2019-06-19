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

