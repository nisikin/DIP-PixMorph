from src.basic.geometry import *
from src.basic.io import load_image, save_image

if __name__ == "__main__":
    img = load_image("../assets/test.jpg")

    # 旋转45度
    rotated = rotate_image(img, 45)

    # 缩放到一半
    resized = resize_image(img, 0.5, 0.5)

    # 平移
    translated = translate_image(img, 100, 50)

    # 镜像翻转
    flipped = flip_image(img, 1)

    # 仿射变换示例
    src_tri = np.float32([[50, 50], [200, 50], [50, 200]])
    dst_tri = np.float32([[10, 100], [200, 50], [100, 250]])
    affine = affine_transform(img, src_tri, dst_tri)

    save_image(rotated, "../assets/rotated.bmp")
    save_image(resized, "../assets/resized.bmp")
    save_image(translated, "../assets/translated.bmp")
    save_image(flipped, "../assets/flipped.bmp")
    save_image(affine, "../assets/affine.bmp")

