import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from src.pixel.utils.detector import EdgeDetectorModule  # 自定义边缘检测模块
from src.pixel.utils.effect import PixelEffectModule    # 自定义像素化效果模块

# 定义主模型：将照片转换为像素风格
class Photo2PixelModel(nn.Module):
    def __init__(self):
        super(Photo2PixelModel, self).__init__()
        self.module_pixel_effect = PixelEffectModule()   # 像素化模块
        self.module_edge_detect = EdgeDetectorModule()   # 边缘检测模块

    def forward(self, rgb,
                param_kernel_size=10,
                param_pixel_size=16,
                param_edge_thresh=112):

        # 1. 先进行像素化处理（可包含模糊 + 下采样等）
        rgb = self.module_pixel_effect(rgb, 4, param_kernel_size, param_pixel_size)

        # 2. 检测图像中的边缘
        edge_mask = self.module_edge_detect(rgb, param_edge_thresh, param_edge_dilate=3)

        # 3. 用边缘掩码把边缘区域“挖空”（设置为0，黑色）
        rgb = torch.masked_fill(rgb, torch.gt(edge_mask, 0.5), 0)

        return rgb


def test1():
    img = Image.open("../test.jpg").convert("RGB")  # 打开图像
    img_np = np.array(img).astype(np.float32)                                # 转为NumPy数组
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]       # 转为 [1, 3, H, W]
    img_pt = torch.from_numpy(img_np)                                        # 转为PyTorch张量

    model = Photo2PixelModel()
    model.eval()  # 设置为评估模式

    with torch.no_grad():  # 关闭梯度计算，加快推理速度
        result_rgb_pt = model(img_pt, param_kernel_size=11, param_pixel_size=16)
        result_rgb_pt = result_rgb_pt[0, ...].permute(1, 2, 0)  # 转为 [H, W, 3]

    print("img_pt", img_pt.shape)
    print("result_rgb_pt", result_rgb_pt.shape)

    # 保存输出图像
    result_rgb_np = result_rgb_pt.cpu().numpy().astype(np.uint8)
    Image.fromarray(result_rgb_np).save("./test_result_photo2pixel.png")

# 主程序入口
if __name__ == '__main__':
    test1()
