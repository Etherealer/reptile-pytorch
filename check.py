import argparse
import os

from PIL import Image, UnidentifiedImageError


def check_image_integrity(file_path):
    try:
        with Image.open(file_path) as img:
            img.load()
    except (UnidentifiedImageError, IOError, OSError):
        os.remove(file_path)
        print(f"Image '{file_path}' is corrupted or not readable.")


def check_images_in_folder(folder_path):
    supported_extension = '.jpeg'
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(supported_extension):
                file_path = os.path.join(root, filename)
                check_image_integrity(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/miniimagenet')
    args = parser.parse_args()
    check_images_in_folder(args.path)
