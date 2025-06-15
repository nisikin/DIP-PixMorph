import subprocess

def run_neural_style_train(dataset, style_image, save_model_dir,
                           epochs=10, batch_size=4, image_size=256, accel=True):
    cmd = [
        "python", "neural_style.py", "train",
        "--dataset", dataset,
        "--style-image", style_image,
        "--save-model-dir", save_model_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--image-size", str(image_size)
    ]
    if accel:
        cmd.append("--accel")
    print("启动训练命令：", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_neural_style_train(
        dataset="../data",
        style_image="../data/style-images/yourname_0.png",
        save_model_dir="../../saved_models/",
        epochs=10,
        batch_size=4,
        image_size=256,
        accel=True
    )
