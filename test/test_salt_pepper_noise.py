from src.basic.salt_pepper_noise import *
import os
from src.basic.io import load_image, save_image

def test_to_grayscale():
    input_path = os.path.join("../assets", "test.jpg")
    output_path = os.path.join("../assets", "test_salt_pepper_noise.jpg")

    img = load_image(input_path)
    gray = add_salt_pepper_noise(img,0.1,0.1)
    save_image(gray, output_path)
    print("添加椒盐噪声测试完成，输出路径：", output_path)

if __name__ == "__main__":
    test_to_grayscale()
