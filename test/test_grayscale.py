import os
from src.basic.io import load_image, save_image
from src.basic.grayscale import grayscale_image

def test_to_grayscale():
    input_path = os.path.join("../assets", "test.jpg")
    output_path = os.path.join("../assets", "test_gray.jpg")

    img = load_image(input_path)
    gray = grayscale_image(img)
    save_image(gray, output_path)
    print("灰度化测试完成，输出路径：", output_path)

if __name__ == "__main__":
    test_to_grayscale()
