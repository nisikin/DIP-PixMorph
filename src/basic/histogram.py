import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_transform(img,weather_gray):
    """
    对图像进行对数变换增强（支持灰度和彩色图）
    """
    if weather_gray  == 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_float = np.float32(img) + 1  # 避免 log(0)
    c = 255 / np.log(1 + np.max(img_float))

    log_img = c * np.log(img_float)
    log_img = np.uint8(np.clip(log_img, 0, 255))
    return log_img

def equalize_gray_histogram(img):
    """
    灰度图像的直方图均衡化
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img_gray)


def equalize_color_histogram(img_bgr):
    """
    彩色图像的直方图均衡化：转换为 YUV，在 Y 通道均衡后再转回
    """
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # Y 通道均衡
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def plot_histogram(img, title="Histogram"):
    """
    画出图像的直方图（灰度或彩色）
    """
    plt.figure(figsize=(8, 4))
    if len(img.shape) == 2:  # 灰度图
        plt.hist(img.ravel(), bins=256, range=[0, 256], color='gray')
        plt.title(f"{title} - Gray")
    else:  # 彩色图
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.title(f"{title} - Color Channels")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def histogram_normalization(img, a=0, b=255):
    """
    对图像进行直方图正规化，把像素值拉伸到 [a, b]
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = img.astype(np.float32)
    min_val, max_val = np.min(img_float), np.max(img_float)
    norm_img = (img_float - min_val) / (max_val - min_val) * (b - a) + a
    return np.uint8(np.clip(norm_img, a, b))

