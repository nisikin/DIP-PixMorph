import subprocess
import sys
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 关闭像素限制警告

def run_neural_style_train(dataset, style_image, save_model_dir,
                           epochs=10, batch_size=1, image_size=128, accel=True):
    python_executable = sys.executable

    # 绝对路径
    dataset = os.path.abspath(dataset)
    style_image = os.path.abspath(style_image)
    save_model_dir = os.path.abspath(save_model_dir)

    # 打印提示，提醒用户注意内存
    print(f"注意：已设置 image_size={image_size} 和 batch_size={batch_size}，以避免内存不足问题。")

    cmd = [
        python_executable,
        os.path.abspath("../neural_style/neural_style.py"),
        "train",
        "--dataset", dataset,
        "--style-image", style_image,
        "--save-model-dir", save_model_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--image-size", str(image_size),
    ]
    if accel:
        cmd.append("--accel")

    print("启动训练命令：", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_neural_style_train(
        dataset="../data",
        style_image="../data/style-images2/disco_0.jpg",
        save_model_dir="../models/",
        epochs=10,
        batch_size=1,     # 减小 batch_size
        image_size=128,   # 减小图片尺寸
        accel=True
    )
