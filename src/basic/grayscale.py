import cv2
import numpy as np

def grayscale_image(image: np.ndarray) -> np.ndarray:
    """将图像转换为灰度图"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
