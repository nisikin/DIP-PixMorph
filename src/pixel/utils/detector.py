import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class EdgeDetectorModule(nn.Module):
    def __init__(self):
        super(EdgeDetectorModule, self).__init__()

        # 使用反射填充以避免边缘信息丢失
        self.pad = nn.ReflectionPad2d(padding=(1, 1, 1, 1))

        # Sobel 水平边缘检测核（3×3）
        kernel_sobel_h = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32).reshape([1, 1, 3, 3])
        kernel_sobel_h = torch.from_numpy(kernel_sobel_h).repeat([3, 1, 1, 1])  # 复制3份用于RGB三通道
        self.conv_h = nn.Conv2d(3, 3, kernel_size=3, padding=0, groups=3, bias=False)  # 分组卷积处理每个通道
        self.conv_h.weight = nn.Parameter(kernel_sobel_h)  # 设置权重为Sobel核

        # Sobel 垂直边缘检测核（3×3）
        kernel_sobel_v = np.array([
            [-1, -2, -1],
            [0,  0,  0],
            [1,  2,  1]
        ], dtype=np.float32).reshape([1, 1, 3, 3])
        kernel_sobel_v = torch.from_numpy(kernel_sobel_v).repeat([3, 1, 1, 1])
        self.conv_v = nn.Conv2d(3, 3, kernel_size=3, padding=0, groups=3, bias=False)
        self.conv_v.weight = nn.Parameter(kernel_sobel_v)

    def forward(self, rgb, param_edge_thresh, param_edge_dilate):

        rgb = self.pad(rgb)  # 先进行边界填充
        edge_h = self.conv_h(rgb)  # Sobel 横向梯度
        edge_w = self.conv_v(rgb)  # Sobel 纵向梯度

        # 求出两个方向的绝对值并取最大值，表示边缘强度
        edge = torch.stack([torch.abs(edge_h), torch.abs(edge_w)], dim=1)
        edge = torch.max(edge, dim=1)[0]

        # 将3通道边缘图转为灰度图（均值）
        edge = torch.mean(edge, dim=1, keepdim=True)

        # 阈值处理，得到二值边缘图
        edge = torch.gt(edge, param_edge_thresh).float()

        # 膨胀操作，扩大边缘区域
        edge = F.max_pool2d(edge, kernel_size=param_edge_dilate, stride=1, padding=param_edge_dilate // 2)
        return edge

def test():
    # 读取图像并转换为张量 [1, 3, H, W]
    rgb = np.array(Image.open("../test.jpg").convert("RGB")).astype(np.float32)
    rgb = torch.from_numpy(rgb).permute([2, 0, 1]).unsqueeze(dim=0)

    # 初始化边缘检测模型并运行
    net = EdgeDetectorModule()
    edge_mask = net(rgb, param_edge_thresh=128, param_edge_dilate=3)
    print(edge_mask.shape)  # 输出边缘掩码形状

    # 保存边缘检测结果为图片
    edge_mask = 255 * edge_mask
    edge_mask = edge_mask[0, ...].permute(1, 2, 0).repeat([1, 1, 3])  # 转换为RGB格式
    edge_mask = edge_mask.cpu().numpy().astype(np.uint8)
    Image.fromarray(edge_mask).save("test_result_edge.png")

# 主程序
if __name__ == '__main__':
    test()
