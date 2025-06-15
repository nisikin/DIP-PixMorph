import subprocess
import sys
import os

def run_neural_style_train(dataset, style_image, save_model_dir,
                           epochs=10, batch_size=4, image_size=256, accel=True):
    python_executable = sys.executable

    # 转成绝对路径，确保路径准确
    dataset = os.path.abspath(dataset)
    style_image = os.path.abspath(style_image)
    save_model_dir = os.path.abspath(save_model_dir)

    cmd = [
        python_executable,
        os.path.abspath("../neural_style/neural_style.py"),  # 同样用绝对路径
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
        style_image="../data/style-images/yourname_0.png",
        save_model_dir="../saved_models/",
        epochs=10,
        batch_size=4,
        image_size=256,
        accel=False  # 设置为 False 则不传 --accel 参数
    )
