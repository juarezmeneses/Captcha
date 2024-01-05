import cv2
from PIL import Image

methods = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV,
]

#imagem = cv2.imread("bdcaptcha/imagem_captcha_300.png")
imagem = cv2.imread("bd_fgts/imagem_captcha_300.png")

# transformar a imagem em escala de cinza

#RGB(255, 0, 0)
#Red - 0 a 255
#Green - 0 a 255
#Blue - 0 a 255

image_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
#image_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

blurred_image = cv2.GaussianBlur(image_gray, (5, 5), 0)

i = 0
for method in methods:
    i += 1
    _, handle_image = cv2.threshold(blurred_image, 127, 255, method or cv2.THRESH_OTSU)
    #_, handle_image = cv2.threshold(image_gray, 0, 255, method + cv2.THRESH_OTSU)
    cv2.imwrite(f'testmethod/handle_image_{i}.png', handle_image)

image = Image.open("testmethod/handle_image_4.png")
image = image.convert("P")
#image2 = Image.new("P", image.size, 255)
image2 = Image.new("P", image.size,(255, 255, 255))

for x in range(image.size[1]):
    for y in range(image.size[0]):
        color_pixel = image.getpixel((y, x))
        print(f"Pixel na posição ({y}, {x}): {color_pixel}")
        if color_pixel > 215:
            #image2.putpixel((y, x), 0)
            image2.putpixel((y, x), (0, 0, 0))

image2.save('testmethod/imagefinal.png')