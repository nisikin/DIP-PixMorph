import cv2
import numpy as np
from scipy import ndimage


def roberts_edge(img):
    """
    Roberts 算子边缘检测
    """
    kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_x = ndimage.convolve(img, kernel_x)
    edge_y = ndimage.convolve(img, kernel_y)
    edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
    return np.uint8(np.clip(edge, 0, 255))


def prewitt_edge(img):
    """
    Prewitt 算子边缘检测
    """
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_x = ndimage.convolve(img, kernel_x)
    edge_y = ndimage.convolve(img, kernel_y)
    edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
    return np.uint8(np.clip(edge, 0, 255))


def sobel_edge(img):
    """
    Sobel 算子边缘检测
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
    return np.uint8(np.clip(edge, 0, 255))


def laplacian_edge(img):
    """
    Laplacian 算子边缘检测
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Laplacian(img_gray, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(edge), 0, 255))


def log_edge(img):
    """
    LoG（Laplacian of Gaussian）边缘检测
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edge = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.uint8(np.clip(np.abs(edge), 0, 255))


def canny_edge(img, threshold1=100, threshold2=200):
    """
    Canny 边缘检测
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(img_gray, threshold1, threshold2)
    return edge

def hough_lines(img, use_probabilistic=False, threshold=100, min_line_length=50, max_line_gap=10):
    """
    霍夫变换直线检测
    :param img: 输入图像（BGR）
    :param use_probabilistic: 是否使用概率霍夫变换
    :param threshold: 累加器阈值（越大检测越少）
    :param min_line_length: 最小直线长度（仅用于概率霍夫）
    :param max_line_gap: 最大允许的线段间隙（仅用于概率霍夫）
    :return: 带检测直线的图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    output = img.copy()

    if use_probabilistic:
        # 概率霍夫变换
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        # 常规霍夫变换
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
        if lines is not None:
            for rho_theta in lines:
                rho, theta = rho_theta[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output