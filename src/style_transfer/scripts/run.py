# src/style_transfer/neural_style/run.py

import re
import torch
from torchvision import transforms
from src.style_transfer.neural_style.transformer_net import TransformerNet
from src.style_transfer.neural_style import utils
import numpy as np
from PIL import Image

def run_style_transfer(content_image, model_path, content_scale=None, device="cpu"):
    """
    对输入图像执行风格迁移。

    参数：
        content_image (PIL.Image 或 np.ndarray): 原图（非路径）
        model_path (str): .pth 风格模型路径
        content_scale (float or None): 图像缩放比例
        device (str): "cpu" 或 "cuda"

    返回：
        np.ndarray: 风格迁移后的图像
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 若是 ndarray 则转为 PIL.Image
    if isinstance(content_image, np.ndarray):
        content_image = Image.fromarray(content_image)

    # 可选缩放
    if content_scale:
        w, h = content_image.size
        content_image = content_image.resize(
            (int(w / content_scale), int(h / content_scale)),  Image.Resampling.LANCZOS
        )

    # 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_tensor = transform(content_image).unsqueeze(0).to(device)

    # 加载模型
    with torch.no_grad():
        model = TransformerNet()
        state_dict = torch.load(model_path)
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        model.load_state_dict(state_dict)
        model.to(device).eval()

        output = model(content_tensor).cpu()

    # 转为 numpy 图像
    output_image = output[0].clamp(0, 255).detach().numpy()
    output_image = output_image.transpose(1, 2, 0).astype('uint8')

    return output_image
