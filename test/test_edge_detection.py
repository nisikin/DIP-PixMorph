from src.basic.edge_detection import *
from src.basic.io import load_image, save_image


if __name__ == "__main__":

    img = load_image("../assets/test.jpg")

    roberts = roberts_edge(img)
    prewitt = prewitt_edge(img)
    sobel = sobel_edge(img)
    laplacian = laplacian_edge(img)
    log = log_edge(img)
    canny = canny_edge(img)
    hough = hough_lines(img, use_probabilistic=True)

    save_image(roberts, "../assets/roberts.jpg")
    save_image(prewitt, "../assets/prewitt.jpg")
    save_image(sobel, "../assets/sobel.jpg")
    save_image(laplacian, "../assets/laplacian.jpg")
    save_image(log, "../assets/log.jpg")
    save_image(canny, "../assets/canny.jpg")
    save_image(hough, "../assets/hough.jpg")