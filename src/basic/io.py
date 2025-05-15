import cv2

def load_image(path):
    """读取图像并转为 numpy 数组"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # 读取彩色图像
    if img is None:
        raise FileNotFoundError(f"图像文件未找到")
    return img

def save_image(image_array, path):
    """保存 numpy 数组为图像"""
    success = cv2.imwrite(path, image_array)  # 保存图像
    if not success:
        raise Exception(f"无法保存图像")