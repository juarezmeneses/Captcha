import cv2
import os
import glob

from PIL import Image


def handle_images(folder_origin, folder_destiny='bd_fgts_clean'):
        files = glob.glob(f"{folder_origin}/*")
        for file in files:
            image = cv2.imread(file)
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

            _, handle_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_TOZERO or cv2.THRESH_OTSU)
            file_name = os.path.basename(file)
            cv2.imwrite(f'{folder_destiny}/{file_name}', handle_image)

        files = glob.glob(f"{folder_destiny}/*")
        for file in files:
            image = Image.open(file)
            image = image.convert("P")
            image2 = Image.new("P", image.size, (255, 255, 255))

            for x in range(image.size[1]):
                for y in range(image.size[0]):
                    color_pixel = image.getpixel((y, x))
                    #print(f"Pixel na posição ({y}, {x}): {color_pixel}")
                    if color_pixel > 215:
                        image2.putpixel((y, x), (0, 0, 0))
            file_name = os.path.basename(file)
            image2.save(f'{folder_destiny}/{file_name}')


if __name__ == "__main__":
    handle_images('bd_fgts')