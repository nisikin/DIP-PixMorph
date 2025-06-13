from src.basic.histogram import *
from src.basic.io import load_image, save_image

if __name__ == "__main__":
    img = load_image("../assets/test.jpg")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 对数变换
    transform_gray_img = log_transform(img,1)
    transform_color_img = log_transform(img,0)

    # 灰度图均衡
    equalized_gray = equalize_histogram(img,1)

    # 彩色图均衡
    equalized_color = equalize_histogram(img,0)

    # 正规化
    norm_img = histogram_normalization(img)

    # 显示结果
    """cv2.imshow("Original Gray", img_gray)
    cv2.imshow("transformed Gray", transform_gray_img)
    cv2.imshow("Equalized Gray", equalized_gray)
    cv2.imshow("Original Color", img)
    cv2.imshow("transformed Color", transform_color_img)
    cv2.imshow("Equalized Color", equalized_color)
    cv2.imshow("Normal Image", norm_img)

    # 画直方图
    plot_histogram(img_gray, "Original Gray")
    plot_histogram(transform_gray_img,"transformed Gray")
    plot_histogram(equalized_gray, "Equalized Gray")
    plot_histogram(img, "Original Color")
    plot_histogram(transform_color_img, "transformed Color")
    plot_histogram(equalized_color, "Equalized Color")
    plot_histogram(norm_img, "Normal Image")
    """
    img = plot_histogram(img)
    save_image(img, "../assets/histogram.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
