import cv2
import numpy as np

def erode_image(image: np.ndarray, kernel_size=(5, 5), iterations=1) -> np.ndarray:
    """图像腐蚀操作"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.erode(image, kernel, iterations=iterations)

def dilate_image(image: np.ndarray, kernel_size=(5, 5), iterations=1) -> np.ndarray:
    """图像膨胀操作"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)

def open_image(image: np.ndarray, kernel_size=(5, 5), iterations=1) -> np.ndarray:
    """图像开运算操作"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

def close_image(image: np.ndarray, kernel_size=(5, 5), iterations=1) -> np.ndarray:
    """图像闭运算操作"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
