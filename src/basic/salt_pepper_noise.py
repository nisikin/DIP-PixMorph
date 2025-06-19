import numpy as np
import cv2

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    向图像添加椒盐噪声
    :param image: 输入图像（numpy array，BGR或灰度）
    :param salt_prob: 椒噪声（白点）比例
    :param pepper_prob: 盐噪声（黑点）比例
    :return: 添加噪声后的图像
    """
    noisy = image.copy()
    h, w = noisy.shape[:2]

    # 添加 salt（白点）
    num_salt = int(h * w * salt_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in noisy.shape[:2]]
    if noisy.ndim == 2:
        noisy[coords[0], coords[1]] = 255
    else:
        noisy[coords[0], coords[1], :] = 255

    # 添加 pepper（黑点）
    num_pepper = int(h * w * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy.shape[:2]]
    if noisy.ndim == 2:
        noisy[coords[0], coords[1]] = 0
    else:
        noisy[coords[0], coords[1], :] = 0

    return noisy

