from src.basic.io import load_image, save_image

def test_io():
    try:
        # 加载图像
        print(f"正在加载图像...")
        img = load_image("../assets/test.jpg")

        # 保存图像为 BMP 格式
        print(f"正在保存图像...")
        save_image(img, "../assets/test.bmp")

        print(f"图像加载并保存成功...")

    except FileNotFoundError as e:
        print(f"文件未找到")

if __name__ == "__main__":
    test_io()
