# src/style_transfer/neural_style/run.py

import re
import torch
from torchvision import transforms
from .transformer_net import TransformerNet
from . import utils

def run_style_transfer(content_image_path, model_path, output_image_path, content_scale=None, device="cpu"):
    """
    对输入图像执行风格迁移。

    参数：
        content_image_path (str): 原图路径
        model_path (str): .pth 风格模型路径
        output_image_path (str): 输出图像保存路径
        content_scale (float or None): 图像缩放比例
        device (str): "cpu" 或 "cuda"

    返回：
        None（直接保存图像）
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 加载并预处理图片
    content_image = utils.load_image(content_image_path, scale=content_scale)
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

        # 推理并保存图像
        output = model(content_tensor).cpu()
        utils.save_image(output_image_path, output[0])

    return output
