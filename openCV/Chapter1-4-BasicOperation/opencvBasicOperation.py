import cv2 #opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np 
# %matplotlib inline 

def imageReadAndSave():
    '''
    图像读取，保存
    :return:
    '''
    img=cv2.imread('cat.jpg')
    print(img)
    cv_show('image',img)
    #图像的显示,也可以创建多个窗口

    img=cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
    #图像的显示,也可以创建多个窗口
    cv_show('image',img)
    #保存
    cv2.imwrite('mycat.png',img)

def imageGray():
    '''
    图像颜色转换
    :return:
    '''
    img = cv2.imread('cat.jpg')
    # 灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show("gray", img_gray)
    # HSV
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv_show("gray", img_gray)

def videoReadAndSave():
    """
    视频读取，取帧，置灰，显示
    :return:
    """
    vc = cv2.VideoCapture('test.mp4')
    # 检查是否打开正确
    open = False
    if vc.isOpened():
        open, frame = vc.read()

    while open:
        ret, frame = vc.read()
        if frame is None:
            break
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('result', gray)
            if cv2.waitKey(20) & 0xFF == 27:
                break
    vc.release()
    cv2.destroyAllWindows()

def catchPartOfImage():
    '''
    截取图像的部分区域
    :return:
    '''
    img = cv2.imread('cat.jpg')
    cat = img[0:50, 0:200]
    cv_show('cat', cat)

def spiltImageRGB():
    img = cv2.imread('cat.jpg')
    b,g,r = cv2.split(img)

    print(b.shape)
    print(g.shape)
    print(r.shape)

    # 只保留R, [0,1,2] 将其它位置0
    cur_img = img.copy()
    cur_img[:, :, 0] = 0
    cur_img[:, :, 1] = 0
    cv_show('OnlyRed', cur_img)

    img1 = cv2.merge((b,g,0))
    cv_show('img1', img1)

def makeBorder():
    '''
    边界填充
    BORDER_REPLICATE：复制法，也就是复制最边缘像素。
    BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
    BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
    BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
    BORDER_CONSTANT：常量法，常数值填充。
    :return:
    '''
    img = cv2.imread('cat.jpg')
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

    import matplotlib.pyplot as plt
    plt.subplot(321), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
    plt.subplot(322), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
    plt.subplot(323), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.subplot(324), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
    plt.subplot(325), plt.imshow(wrap, 'gray'), plt.title('WRAP')
    plt.subplot(326), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

    plt.show()

def resize():
    '''
    调整大小
    :return:
    '''
    img_dog = cv2.imread('dog.jpg')
    # 按绝对值调整
    img_dog = cv2.resize(img_dog, (500, 414))
    cv_show('resize_dog', img_dog)
    # 按比例调整
    img_dog = cv2.resize(img_dog, (0,0), fx=2, fy=1)
    cv_show('resize_dog', img_dog)

def mergeImage():
    '''
    猫狗融合

    :return:
    '''
    img_cat = cv2.imread('cat.jpg')
    img_dog = cv2.imread('dog.jpg')
    if(img_cat.shape != img_dog.shape):
        print("两个图像大小不等，不能融合，1：{}；2：{}".format(img_cat.shape, img_dog.shape))
        img_dog = cv2.resize(img_dog, (500, 414))
        print("修改图2的大小为：{}".format(img_dog.shape))
    res = cv2.addWeighted(img_cat, 0.6, img_dog, 0.4, 0)
    import matplotlib.pyplot as plt
    plt.subplot(111), plt.imshow(res), plt.title('cat&dog')
    plt.show()

def thresholdHandle():
    '''
    阈值处理
    - cv2.THRESH_BINARY     超过阈值部分取maxval（最大值），否则取0
    - cv2.THRESH_BINARY_INV   THRESH_BINARY的反转
    - cv2.THRESH_TRUNC    大于阈值部分设为阈值，否则不变
    - cv2.THRESH_TRIANGLE
    - cv2.THRESH_TOZERO   大于阈值部分不改变，否则设为0
    - cv2.THRESH_TOZERO_INV    THRESH_TOZERO的反转
    :return:
    '''
    img_cat = cv2.imread('cat.jpg')
    # 灰度
    img_gray = cv2.cvtColor(img_cat, cv2.COLOR_BGR2GRAY)
    ret, thresh0 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRIANGLE)
    ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'THRESH_TRIANGLE', 'TOZERO', 'TOZERO_INV']
    images = [img_cat, thresh0 ,thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(7):
        plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def imageBlur():
    '''
    图像滤波
    :return:
    '''
    img = cv2.imread('lenaNoise.png')
    # 均值滤波：简单的平均卷积操作
    blur = cv2.blur(img, (3, 3))
    # 方框滤波：基本和均值一样，可以选择归一化
    box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
    # 高斯滤波：
    # 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
    aussian = cv2.GaussianBlur(img, (5, 5), 1)
    # 中值滤波
    # 相当于用中值代替
    median = cv2.medianBlur(img, 5)  # 中值滤波
    # 展示所有的
    res = np.hstack((img, blur, box, aussian, median))
    cv_show("VS", res)

def erode():
    '''
    形态 腐蚀操作
    :return:
    '''
    img = cv2.imread('dige.png')
    # 选一个核心区域做腐蚀处理，类似处理的颗粒度。3*3像素大小
    kernel = np.ones((3, 3), np.uint8)
    # iterations 1次腐蚀
    erosion = cv2.erode(img, kernel, iterations=1)
    res = np.hstack((img, erosion))
    cv_show("VS", res)

    # 腐蚀一个圆，多次腐蚀的效果对比
    pie = cv2.imread('pie.png')
    kernel = np.ones((10, 10), np.uint8)
    erosion_1 = cv2.erode(pie, kernel, iterations=1)
    erosion_2 = cv2.erode(pie, kernel, iterations=2)
    erosion_3 = cv2.erode(pie, kernel, iterations=3)
    res = np.hstack((pie, erosion_1, erosion_2, erosion_3))
    cv_show('res', res)

def dilate():
    '''
    形态 膨胀操作
    '''
    img = cv2.imread('dige.png')
    kernel = np.ones((3, 3), np.uint8)
    dige_erosion = cv2.erode(img, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    dige_dilate = cv2.dilate(dige_erosion, kernel, iterations=1)
    res = np.hstack((img, dige_erosion, dige_dilate))
    cv_show('res', res)

    # 膨胀一个圆，多次膨胀的效果对比
    pie = cv2.imread('pie.png')
    kernel = np.ones((30, 30), np.uint8)
    dilate_1 = cv2.dilate(pie, kernel, iterations=1)
    dilate_2 = cv2.dilate(pie, kernel, iterations=2)
    dilate_3 = cv2.dilate(pie, kernel, iterations=3)
    res = np.hstack((dilate_1, dilate_2, dilate_3))
    cv_show('res', res)

def gradient():
    '''
    图像梯度
    计算出边缘变化，类似求导
    Sobel算子
    x:左减右（有权重），   y：下减上
     -1 0 +1           -1 +2 -1
     -2 0 +2            0  0  0
     -1 0 +1           +1 +2 +1
    谁减谁，顺序不重要，因为后面要取绝对值
    求出 x轴、y轴的变化梯度，再相加
    Scharr,类似Sobel，但权重更大，效果上更加敏感，容易有噪音
    x:左减右（有权重），   y：下减上
     -3 0 +3           -3 +10 -3
     -10 0 +10          0  0  0
     -3 0 +3           +3 +10 +3
    Laplacian，中间点，与上下左右4个位置对比的差值，不需要分xy轴了
     0  1  0
     1 -4  1
     0  1  1
    :return:
    '''
    img = cv2.imread('pie.png', cv2.IMREAD_GRAYSCALE)
    # ddepth:图像的深度
    # dx和dy分别表示水平和竖直方向
    # ksize是Sobel算子的大小
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # 白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以要取绝对值
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    # 分别计算x和y，再求和
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    res = np.hstack((img, sobelx, sobely, sobelxy))
    cv_show('res', res)

    # 演示：人像图，获取轮廓
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    res = np.hstack((img, sobelx, sobely, sobelxy))
    cv_show('res', res)

    # 对比其他算子
    # Scharr
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.convertScaleAbs(scharry)
    scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    # Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    res = np.hstack((img, sobelxy, scharrxy, laplacian))
    cv_show('res', res)

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    gradient()