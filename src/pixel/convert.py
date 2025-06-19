import torch
from PIL import Image
import numpy as np

from src.pixel.utils.pixel import Photo2PixelModel
from src.pixel.utils import basic

def convert_to_pixel(
    img_input,
    output_path: str = None,
    kernel_size: int = 8,
    pixel_size: int = 5,
    edge_thresh: int = 150,
    model: Photo2PixelModel = None
) -> Image.Image:
    """
    将照片转换为像素风格图像

    参数：
    - img_input: 输入图像，支持PIL.Image或numpy.ndarray
    - output_path: 可选，输出图像保存路径，若为 None 则不保存
    - kernel_size: 模糊程度
    - pixel_size: 像素块大小
    - edge_thresh: 边缘检测阈值
    - model: 可选，自定义模型

    返回：
    - 处理后的 PIL 图像对象
    """
    # 如果是 numpy.ndarray，转成 PIL.Image
    if isinstance(img_input, np.ndarray):
        img_input = Image.fromarray(img_input)
    elif not isinstance(img_input, Image.Image):
        raise TypeError("img_input 必须是 PIL.Image 或 numpy.ndarray")

    img_pt_input = basic.convert_image_to_tensor(img_input)

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

    if output_path:
        img_output.save(output_path)

    return img_output

def test():
    img_input = Image.open('./test.jpg')
    output_path = './result.jpg'

    img_output = convert_to_pixel(
        img_input,
        output_path=output_path,
    )
    img_output.show()  # 显示结果图像

if __name__ == '__main__':
    test()
