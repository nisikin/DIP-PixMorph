from src.basic.morph_ops import *
import os
from src.basic.io import load_image, save_image

def test_to_grayscale():
    input_path = os.path.join("../assets", "test.jpg")

    img = load_image(input_path)
    erode = erode_image(img)
    dilated = dilate_image(img)
    opened = open_image(img)
    closed = close_image(img)
    save_image(erode, "../assets/test_erode.jpg")
    save_image(dilated, "../assets/test_dilated.jpg")
    save_image(opened, "../assets/test_opened.jpg")
    save_image(closed, "../assets/test_closed.jpg")


if __name__ == "__main__":
    test_to_grayscale()
