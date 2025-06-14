from src.basic.io import load_image, save_image
from src.basic.filtering import *

if __name__ == "__main__":


    img = load_image("../assets/test.jpg")


    mean_img = mean_filter(img)
    median_img = median_filter(img)
    ideal_img = ideal_low_pass_filter(img, cutoff=30)
    butter_img = butterworth_low_pass_filter(img, cutoff=30, order=2)
    gauss_img = gaussian_low_pass_filter(img, cutoff=30)

    save_image(mean_img, "../assets/mean_filter.jpg")
    save_image(median_img, "../assets/median_filter.jpg")
    save_image(ideal_img, "../assets/ideal_low_pass_img.jpg")
    save_image(butter_img, "../assets/butter_smoothing.jpg")
    save_image(gauss_img, "../assets/gauss_smoothing.jpg")

    roberts_img = roberts_sharpen(img)
    sobel_img = sobel_sharpen(img)
    prewitt_img = prewitt_sharpen(img)
    laplacian_img = laplacian_sharpen(img)
    ideal_high_pass_img = ideal_high_pass_filter(img, cutoff=30)
    butter_img = butterworth_high_pass_filter(img, cutoff=30, order=2)
    gauss_img = gaussian_high_pass_filter(img, cutoff=30)

    save_image(roberts_img, "../assets/roberts_sharpen.jpg")
    save_image(sobel_img, "../assets/sobel_sharpen.jpg")
    save_image(prewitt_img, "../assets/prewitt_sharpen.jpg")
    save_image(laplacian_img, "../assets/laplacian_sharpen.jpg")
    save_image(ideal_high_pass_img, "../assets/ideal_high_pass_img.jpg")
    save_image(butter_img, "../assets/butter_sharpen.jpg")
    save_image(gauss_img, "../assets/gauss_sharpen.jpg")

