import cv2
import os
import glob

from PIL import Image

# Pasta onde estão os arquivos de imagem
files = glob.glob('bd_fgts_clean/*')

# Largura esperada para uma única letra ou número
expected_width_of_a_letter = 1  # Ajuste conforme necessário

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
        # 30
        if area > 26:
            area_letters.append((x, y, w, h))
    # if len(area_letters) != 5:
    #     print(f"Pulando arquivo: {file} - Não foi possível encontrar 5 letras ou números")
    #     continue

    image_end = cv2.merge([image] * 3)

    i = 0
    for crop in area_letters:
        x, y, w, h = crop

        # Verifica se a largura é significativamente maior do que o esperado
        print("Largura W:", w)
        if w > 25:
            print("Entrou W:", w)
            # Divide a região identificada em duas partes
            w1 = w // 2
            w2 = w - w1
            image_letter1 = image[y-2:y+h+2, x-2:x+w1+2]
            image_letter2 = image[y-2:y+h+2, x+w1-2:x+w+2]

            i += 1
            file_name1 = os.path.basename(file).replace(".png", f"letter_or_number_{i}_1.png")
            file_name2 = os.path.basename(file).replace(".png", f"letter_or_number_{i}_2.png")

            # Salva a primeira parte
            if image_letter1 is not None and image_letter1.size != 0:
                print("Entrou2 W:", w)
                cv2.imwrite(f'letters_and_numbers/{file_name1}', image_letter1)
            else:
                print(f"Erro: Imagem está vazia. Pulando escrita para o arquivo: {file}")

            # Salva a segunda parte
            if image_letter2 is not None and image_letter2.size != 0:
                print("Entrou3 W:", w)
                cv2.imwrite(f'letters_and_numbers/{file_name2}', image_letter2)
            else:
                print(f"Erro: Imagem está vazia. Pulando escrita para o arquivo: {file}")

        else:
            # Se a largura não for muito grande, proceda como antes
            image_letter = image[y-2:y+h+2, x-2:x+w+2]
            i += 1
            file_name = os.path.basename(file).replace(".png", f"letra_ou_numero_{i}.png")
            if image_letter is not None and image_letter.size != 0:
                cv2.imwrite(f'letters_and_numbers/{file_name}', image_letter)
            else:
                print(f"Erro: Imagem está vazia. Pulando escrita para o arquivo: {file}")

        # Desenha um retângulo para a região identificada
        cv2.rectangle(image_end, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 0), 1)

    file_name = os.path.basename(file)
    cv2.imwrite(f"identified/{file_name}", image_end)

print("Processamento concluído.")