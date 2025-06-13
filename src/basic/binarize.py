import cv2
import numpy as np

def binarize_image(image: np.ndarray, threshold: int = 127,weather_inv: int = 0):
    """将图像转换为灰度图并进行二值化和反二值化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if weather_inv == 0:
        _,binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary
    elif weather_inv == 1:
        _,binary_inv = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        return binary_inv
