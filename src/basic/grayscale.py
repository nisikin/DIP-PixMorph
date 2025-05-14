import cv2
import numpy as np

def image_grayscale(img):
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("输入的图像数据无效，可能是路径错误或格式不支持")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)