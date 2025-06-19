import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class PixelEffectModule(nn.Module):
    def __init__(self):
        super(PixelEffectModule, self).__init__()

    def create_mask_by_idx(self, idx_z, max_z):
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        mask = torch.zeros([h, w, max_z])
        mask[idx_x, idx_y, idx_z] = 1
        return mask

    def select_by_idx(self, data, idx_z):
        h, w = idx_z.shape
        idx_x = torch.arange(h).view([h, 1]).repeat([1, w])
        idx_y = torch.arange(w).view([1, w]).repeat([h, 1])
        return data[idx_x, idx_y, idx_z]

    def forward(self, rgb, param_num_bins, param_kernel_size, param_pixel_size):
        # 拆分RGB通道
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        # 计算每个像素的亮度索引（按平均值划分 bin）
        intensity_idx = torch.mean(rgb, dim=[0, 1]) / 256. * param_num_bins
        intensity_idx = intensity_idx.long()  # [H, W]

        # 构造 one-hot 掩码，按亮度bin划分图像
        intensity = self.create_mask_by_idx(intensity_idx, max_z=param_num_bins)  # [H, W, B]
        intensity = torch.permute(intensity, dims=[2, 0, 1]).unsqueeze(dim=0)      # [1, B, H, W]

        # 对 RGB 通道进行按 bin 加权
        r, g, b = r * intensity, g * intensity, b * intensity

        # 定义平均模糊卷积核，分组卷积保持每个 bin 独立处理
        kernel_conv = torch.ones([param_num_bins, 1, param_kernel_size, param_kernel_size])

        # 对每个bin图像块下采样并模糊处理，输出 [1, B, h, w]
        r = F.conv2d(r, kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size,
                     groups=param_num_bins)[0]
        g = F.conv2d(g, kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size,
                     groups=param_num_bins)[0]
        b = F.conv2d(b, kernel_conv, padding=(param_kernel_size - 1) // 2, stride=param_pixel_size,
                     groups=param_num_bins)[0]
        intensity = F.conv2d(intensity, kernel_conv, padding=(param_kernel_size - 1) // 2,
                             stride=param_pixel_size, groups=param_num_bins)[0]

        # 找出每个像素最强亮度bin
        intensity_max, intensity_argmax = torch.max(intensity, dim=0)  # [h, w]

        # 将 [B, h, w] → [h, w, B]，按最大bin索引选出RGB值
        r = torch.permute(r, dims=[1, 2, 0])
        g = torch.permute(g, dims=[1, 2, 0])
        b = torch.permute(b, dims=[1, 2, 0])
        r = self.select_by_idx(r, intensity_argmax)
        g = self.select_by_idx(g, intensity_argmax)
        b = self.select_by_idx(b, intensity_argmax)

        # 恢复RGB比例（防止强度压缩）
        r = r / intensity_max
        g = g / intensity_max
        b = b / intensity_max

        # 合并RGB并恢复为 [1, 3, H, W] 格式
        result_rgb = torch.stack([r, g, b], dim=-1)
        result_rgb = torch.permute(result_rgb, dims=[2, 0, 1]).unsqueeze(dim=0)

        # 使用上采样恢复原图分辨率（像素块放大）
        result_rgb = F.interpolate(result_rgb, scale_factor=param_pixel_size)

        return result_rgb

def test1():
    img = Image.open("../test.jpg").convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]  # [1, 3, H, W]
    img_pt = torch.from_numpy(img_np)

    model = PixelEffectModule()
    model.eval()

    with torch.no_grad():
        result_rgb_pt = model(img_pt, param_num_bins=4, param_kernel_size=11, param_pixel_size=16)
        result_rgb_pt = result_rgb_pt[0, ...].permute(1, 2, 0)  # [H, W, 3]

    print("img_pt", img_pt.shape)
    print("result_rgb_pt", result_rgb_pt.shape)

    result_rgb_np = result_rgb_pt.cpu().numpy().astype(np.uint8)
    Image.fromarray(result_rgb_np).save("./test_result_pixel_effect.png")

if __name__ == '__main__':
    test1()
