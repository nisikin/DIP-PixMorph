import cv2
import numpy as np


def resize_image(img, scale_x=1.0, scale_y=1.0, interpolation=cv2.INTER_LINEAR):
    """
    图像缩放
    :param scale_x: x方向缩放因子
    :param scale_y: y方向缩放因子
    :param interpolation:
    """
    return cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=interpolation)


def rotate_image(img, angle, center=None, scale=1.0):
    """
    图像旋转
    :param angle: 旋转角度（逆时针）
    :param center: 旋转中心（默认是图像中心）
    :param scale: 缩放因子
    """
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def translate_image(img, tx, ty):
    """
    图像平移
    :param tx: x 方向平移
    :param ty: y 方向平移
    """
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    (h, w) = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def flip_image(img, flip_code):
    """
    图像翻转
    :param flip_code: 0 = 垂直翻转, 1 = 水平翻转, -1 = 水平+垂直翻转
    """
    return cv2.flip(img, flip_code)


def affine_transform(img, src_points, dst_points):
    """
    仿射变换
    :param src_points: 源图像中的三个点（numpy数组，形如 [[x1, y1], [x2, y2], [x3, y3]]）
    :param dst_points: 目标图像中的三个点
    """
    M = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))
    (h, w) = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


