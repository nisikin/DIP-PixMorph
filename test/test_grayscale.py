from src.basic.grayscale import image_grayscale
from src.basic.io import load_image, save_image

def test_grayscale():
    try:
        # 加载图像
        print(f"正在加载图像...")
        img = load_image("../assets/test.jpg")

        # 图像灰度化
        img = image_grayscale(img)
        print(f"图像灰度化成功...")

        # 保存图像为 BMP 格式
        print(f"正在保存图像...")
        save_image(img, "../assets/test.bmp")

        print(f"图像加载并保存成功...")

    except FileNotFoundError as e:
        print(f"文件未找到")
    except Exception as e:
        print(f"发生错误")

if __name__ == "__main__":
    test_grayscale()