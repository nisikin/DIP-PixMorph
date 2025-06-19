import torch
import numpy as np
from PIL import Image

# 将PIL图片转换为PyTorch张量
def convert_image_to_tensor(img):
    img = img.convert("RGB")  # 确保图像为RGB模式（3通道）
    img_np = np.array(img).astype(np.float32)  # 转为NumPy数组，并转换为float32类型
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    # 将图像从(H, W, C)转为(N, C, H, W)，其中N=1表示batch维度
    img_pt = torch.from_numpy(img_np)  # 转换为PyTorch张量
    return img_pt

# 将PyTorch张量还原为PIL图片
def convert_tensor_to_image(img_pt):
    img_pt = img_pt[0, ...].permute(1, 2, 0)  # 去掉batch维度，并转为(H, W, C)顺序
    result_rgb_np = img_pt.cpu().numpy().astype(np.uint8)  # 转为NumPy数组，并转换为uint8格式
    return Image.fromarray(result_rgb_np)  # 转换为PIL图像返回
