import cv2
import os
import glob

from PIL import Image

files = glob.glob('bd_fgts_clean/*')
for file in files:
    print(f"Processando arquivo: {file}")
    image = cv2.imread(file)
    if image is None:
        print(f"Não foi possível ler a imagem: {file}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, new_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area_letters = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 30:
            area_letters.append((x, y, w, h))
    if len(area_letters) != 5:
        print(f"Pulando arquivo: {file} - Não encontrou 5 letras ou números")
        continue

    image_end = cv2.merge([image] * 3)

    i = 0
    for crop in area_letters:
        x, y, w, h = crop
        image_letter = image[y-2:y+h+2, x-2:x+w+2]
        i += 1
        file_name = os.path.basename(file).replace(".png", f"letter_or_number_{i}.png")
        cv2.imwrite(f'letters_and_numbers/{file_name}', image_letter)
        cv2.rectangle(image_end, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 1)

    file_name = os.path.basename(file)
    cv2.imwrite(f"identified/{file_name}", image_end)

print("Processamento concluído.")