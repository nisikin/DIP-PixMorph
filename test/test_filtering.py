from src.basic.io import load_image, save_image
from src.basic.filtering import ideal_low_pass_filter,butterworth_low_pass_filter,gaussian_low_pass_filter,mean_filter,median_filter

if __name__ == "__main__":


    img = load_image("../assets/test.jpg")


    mean_img = mean_filter(img)
    median_img = median_filter(img)
    ideal_img = ideal_low_pass_filter(img, cutoff=30)
    butter_img = butterworth_low_pass_filter(img, cutoff=30, order=2)
    gauss_img = gaussian_low_pass_filter(img, cutoff=30)

    save_image(mean_img, "../assets/mean_img.jpg")
    save_image(median_img, "../assets/median_img.jpg")
    save_image(ideal_img, "../assets/ideal_img.jpg")
    save_image(butter_img, "../assets/butter_img.jpg")
    save_image(gauss_img, "../assets/gauss_img.jpg")
