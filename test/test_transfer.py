from src.style_transfer.neural_style.run import run_style_transfer
from PIL import Image
import os

if __name__ == "__main__":
    # 1. 读取图像
    content_image = Image.open("../assets/test.jpg")

    # 2. 模型路径
    model_path = "../src/style_transfer/saved_models/udnie.pth"

    # 3. 执行风格迁移
    output_image = run_style_transfer(
        content_image,
        model_path=model_path,
        content_scale=1.0,
        device="cpu"
    )

    # 4. 保存结果
    from PIL import Image
    Image.fromarray(output_image).save("../assets/trans_output.jpg")
