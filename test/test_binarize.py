import os
from src.basic.io import load_image, save_image
from src.basic.binarize import binarize_image

def test_binarize_image():
    input_path = os.path.join("../assets", "test.jpg")
    output_bin = os.path.join("../assets", "test_binary.jpg")
    output_inv = os.path.join("../assets", "test_binary_inv.jpg")

    img = load_image(input_path)
    binary, binary_inv = binarize_image(img)

    save_image(binary, output_bin)
    save_image(binary_inv, output_inv)
    print("二值化测试成功")

if __name__ == "__main__":
    test_binarize_image()
