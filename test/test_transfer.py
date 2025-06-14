from src.style_transfer.neural_style.run import run_style_transfer
import sys
import os

if __name__ == "__main__":
    run_style_transfer(
        content_image_path="../assets/test.jpg",

        model_path="../src/style_transfer/saved_models/udnie.pth",

        output_image_path="../assets/trans_output.jpg",

        content_scale=1.0,

        device="cpu"
    )
