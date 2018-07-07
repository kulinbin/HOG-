import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def wise_element_sum(img, fil):
    res = (img * fil).sum()
    if (res < 0):
        res = 0
    elif res > 255:
        res = 255
    return res

def convolve(img, fil):

    fil_heigh = fil.shape[0]  # 获取卷积核(滤波)的高度
    fil_width = fil.shape[1]  # 获取卷积核(滤波)的宽度

    conv_heigh = img.shape[0] - fil.shape[0] + 1  # 确定卷积结果的大小
    conv_width = img.shape[1] - fil.shape[1] + 1

    conv = np.zeros((conv_heigh, conv_width), dtype='uint8')

    for i in range(conv_heigh):
        for j in range(conv_width):  # 逐点相乘并求和得到每一个点
            conv[i][j] = wise_element_sum(img[i:i + fil_heigh, j:j + fil_width], fil)
    return conv

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def Hog(path):
    img = mpimg.imread(path)
    img = rgb2gray(img)
    mysize = img.shape

    img = np.sqrt(img)
    fy = np.array([[-1], [0], [1]])
    fx = np.array([[-1, 0, 1]])
    Ix=convolve(img,fx)
    Iy=convolve(img,fy)
    xx=np.zeros([Ix.shape[0],Iy.shape[1]-Ix.shape[1]])
    Ix=np.column_stack((Ix,xx))
    yy = np.zeros([Ix.shape[0] - Iy.shape[0],Iy.shape[1]])
    Iy = np.row_stack((Iy,yy))

    plt.imshow(Iy)
    plt.show()

Hog('k.jpg')
