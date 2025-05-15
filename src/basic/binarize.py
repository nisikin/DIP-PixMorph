import cv2
import numpy as np

def binarize_image(image: np.ndarray, threshold: int = 127) -> tuple[np.ndarray, np.ndarray]:
    """将图像转换为灰度图并进行二值化和反二值化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    _, binary_inv = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary, binary_inv
