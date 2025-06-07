import cv2
import numpy as np

"""
图像的平滑包括空域的平滑和频域的平滑
其中空域的平滑是直接在图像的像素值上进行操作，包括均值滤波和中值滤波。
频域的平滑是将图像通过傅里叶变换转换到频域，然后对频率成分进行操作，再反变换回图像空间，包括理想_低通滤波，巴特沃斯低通滤波和高斯低通滤波。
"""


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """均值滤波"""
    return cv2.blur(image, (kernel_size, kernel_size))

def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """中值滤波（对椒盐噪声效果好）"""
    return cv2.medianBlur(image, kernel_size)

def ideal_low_pass_filter(image, cutoff):
    """理想低通滤波器"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # DFT + 中心化
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 构造理想低通掩模
    mask = np.zeros((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            if np.sqrt((u - crow)**2 + (v - ccol)**2) <= cutoff:
                mask[u, v] = 1

    # 频域滤波
    filtered = dft_shift * mask

    # 反DFT
    f_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def butterworth_low_pass_filter(image, cutoff, order=2):
    """巴特沃斯低通滤波器"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            H = 1 / (1 + (D / cutoff) ** (2 * order))
            mask[u, v] = H

    filtered = dft_shift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def gaussian_low_pass_filter(image, cutoff):
    """高斯低通滤波器"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((rows, cols, 2), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow)**2 + (v - ccol)**2)
            H = np.exp(-(D ** 2) / (2 * (cutoff ** 2)))
            mask[u, v] = H

    filtered = dft_shift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

