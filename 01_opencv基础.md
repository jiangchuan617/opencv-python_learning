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

