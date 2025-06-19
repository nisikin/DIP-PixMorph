import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.pixel.utils.pixel import Photo2PixelModel
from src.pixel.utils import basic


def adaptive_pixel_border(image, pixel_size, border_params):
    """
    自适应像素边框添加功能
    根据图像区域特征智能调整边框颜色和宽度
    """
    # 创建用于绘制的图像副本
    bordered = image.copy()
    draw = ImageDraw.Draw(bordered)
    width, height = image.size

    # 创建图像的小尺寸版本用于快速颜色分析
    small_img = image.resize((width // 10, height // 10), Image.NEAREST)

    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            # 计算当前块区域
            x_end = min(x + pixel_size, width)
            y_end = min(y + pixel_size, height)

            # 获取当前块的主色
            region = image.crop((x, y, x_end, y_end))
            region_color = np.array(region).mean(axis=(0, 1)).astype(int)

            # 获取周边区域主色（用于边缘检测）
            center_x = min(x + pixel_size // 2, width - 1)
            center_y = min(y + pixel_size // 2, height - 1)

            # 安全索引缩小图像坐标
            sx = min(center_x // 10, small_img.width - 1)
            sy = min(center_y // 10, small_img.height - 1)
            neighbor_color = np.array(small_img.getpixel((sx, sy)))

            # 自适应边框颜色
            if border_params['adaptive']:
                # 计算亮度差异
                color_diff = np.abs(np.array(region_color) - neighbor_color).mean()

                # 根据颜色差异调整边框
                if color_diff < 30:  # 类似颜色区域
                    border_color = tuple(np.clip(region_color - 15, 0, 255))
                    border_width = max(1, border_params['width'] // 2)
                else:  # 高对比度区域
                    border_color = tuple(np.clip(region_color - 40, 0, 255))
                    border_width = border_params['width']
            else:
                border_color = border_params['color']
                border_width = border_params['width']

            # 避免边框过宽
            max_allowed = max(1, pixel_size // 3)
            border_width = min(border_width, max_allowed)

            # 安全绘制边框
            if border_width > 0:
                # 绘制填充矩形
                draw.rectangle([x, y, x_end, y_end], fill=tuple(region_color))

                # 绘制边框（只绘制在非边缘位置）
                if x_end < width:
                    draw.line([(x_end, y), (x_end, y_end)], fill=border_color, width=border_width)
                if y_end < height:
                    draw.line([(x, y_end), (x_end, y_end)], fill=border_color, width=border_width)

    return bordered


def convert_to_pixel(
        img_input,
        output_path: str = None,
        kernel_size: int = 10,
        pixel_size: int = 16,
        edge_thresh: int = 100,
        model: Photo2PixelModel = None,
        border_style: dict = None
) -> Image.Image:
    """
    参数：
    - img_input: 输入图像，支持PIL.Image或numpy.ndarray
    - output_path: 可选，输出图像保存路径
    - kernel_size: 模糊程度
    - pixel_size: 像素块大小
    - edge_thresh: 边缘检测阈值
    - model: 可选，自定义模型
    - border_style: 边框风格配置
        adaptive: 是否启用自适应边框 (True/False)
        width: 边框宽度 (像素)
        color: 边框颜色 (RGB元组)

    返回：
    - 处理后的 PIL 图像对象
    """
    # 输入验证和转换
    if isinstance(img_input, np.ndarray):
        img_pil = Image.fromarray(img_input)
    elif isinstance(img_input, Image.Image):
        img_pil = img_input.copy()
    else:
        raise TypeError("img_input 必须是 PIL.Image 或 numpy.ndarray")

    # 使用模型进行像素化处理
    img_pt_input = basic.convert_image_to_tensor(img_pil)

    if model is None:
        model = Photo2PixelModel()
    model.eval()

    with torch.no_grad():
        img_pt_output = model(
            img_pt_input,
            param_kernel_size=kernel_size,
            param_pixel_size=pixel_size,
            param_edge_thresh=edge_thresh
        )

    img_output = basic.convert_tensor_to_image(img_pt_output)

    # 设置默认边框风格
    if border_style is None:
        border_style = {
            'adaptive': True,
            'width': 1,
            'color': (30, 30, 30)
        }
    elif 'adaptive' not in border_style:
        border_style['adaptive'] = True

    # 应用智能边框效果
    if border_style['width'] > 0:
        img_output = adaptive_pixel_border(img_output, pixel_size, border_style)

    # 微调整体对比度
    img_output = Image.fromarray(np.array(img_output))
    img_output = img_output.filter(ImageFilter.UnsharpMask(
        radius=0.5,
        percent=150,
        threshold=3
    ))

    # 保存结果
    if output_path:
        img_output.save(output_path)

    return img_output


def test():
    # 测试图像路径
    img_input = Image.open('../../assets/mylifewill.jpg')
    output_path = './result.jpg'

    img_output = convert_to_pixel(
        img_input,
        output_path=output_path,
    )

    # 显示结果
    img_output.show()

if __name__ == '__main__':
    test()