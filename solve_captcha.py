import glob
import os

from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import cv2
import pickle
from handle_captcha import handle_images

def solve_captcha():
    with open("labels_model.dat", "rb") as translator_file:
        lb = pickle.load(translator_file)

    model = load_model("trained_model.hdf5")

    handle_images("solve", folder_destiny="solve")

##############
    #files = glob.glob('bd_fgts_clean/*')
    files = list(paths.list_images("solve"))
    for file in files:
        #print(f"Processando arquivo: {file}")
        image = cv2.imread(file)
        if image is None:
            #print(f"Não foi possível ler a imagem: {file}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, new_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_letters = []

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            #30
            #26
            if area > 22:
                area_letters.append((x, y, w, h))

        area_letters = sorted(area_letters, key=lambda list_touple: list_touple[0])
        #if len(area_letters) != 5:
        #    print(f"Pulando arquivo: {file} - Não encontrou 5 letras ou números")
        #    continue

        image_end = cv2.merge([image] * 3)
        preview = []

        i = 0
        for crop in area_letters:
            x, y, w, h = crop
            # print("Entrou W:", w)
            #28
            if w > 25:
                # print("Entrou W2:", w)
                # Divide a região identificada em duas partes
                w1 = w // 2
                w2 = w - w1
                image_letter1 = image[y - 2:y + h + 2, x - 2:x + w1 + 2]
                image_letter2 = image[y - 2:y + h + 2, x + w1 - 2:x + w + 2]

                image_letter1 = resize_to_fit(image_letter1, 20, 20)
                image_letter2 = resize_to_fit(image_letter2, 20, 20)

                image_letter1 = np.expand_dims(image_letter1, axis=2)
                image_letter1 = np.expand_dims(image_letter1, axis=0)
                image_letter2 = np.expand_dims(image_letter2, axis=2)
                image_letter2 = np.expand_dims(image_letter2, axis=0)

                preview_letter1 = model.predict(image_letter1)
                preview_letter1 = lb.inverse_transform(preview_letter1)[0]
                preview_letter1 = preview_letter1.replace("_upper", "").replace("_lower", "")

                preview.append(preview_letter1)

                preview_letter2 = model.predict(image_letter2)
                preview_letter2 = lb.inverse_transform(preview_letter2)[0]
                preview_letter2 = preview_letter2.replace("_upper", "").replace("_lower", "")

                preview.append(preview_letter2)

                # i += 1
                # file_name1 = os.path.basename(file).replace(".png", f"letter_or_number_{i}_1.png")
                # file_name2 = os.path.basename(file).replace(".png", f"letter_or_number_{i}_2.png")
                #
                # # Salva a primeira parte
                # if image_letter1 is not None and image_letter1.size != 0:
                #     print("Entrou2 W:", w)
                #     cv2.imwrite(f'letters_and_numbers/{file_name1}', image_letter1)
                # else:
                #     print(f"Erro: Imagem está vazia. Pulando escrita para o arquivo: {file}")
                #
                # # Salva a segunda parte
                # if image_letter2 is not None and image_letter2.size != 0:
                #     print("Entrou3 W:", w)
                #     cv2.imwrite(f'letters_and_numbers/{file_name2}', image_letter2)
                # else:
                #     print(f"Erro: Imagem está vazia. Pulando escrita para o arquivo: {file}")

            else:
                image_letter = image[y - 2:y + h + 2, x - 2:x + w + 2]
                #i += 1
                #file_name = os.path.basename(file).replace(".png", f"letter_or_number_{i}.png")
                #cv2.imwrite(f'letters_and_numbers/{file_name}', image_letter)

                image_letter = resize_to_fit(image_letter, 20, 20)

                image_letter = np.expand_dims(image_letter, axis=2)
                image_letter = np.expand_dims(image_letter, axis=0)

                preview_letter = model.predict(image_letter)

                #matrix decode
                preview_letter = lb.inverse_transform(preview_letter)[0]

                preview_letter = preview_letter.replace("_upper", "").replace("_lower", "")

                preview.append(preview_letter)

                #cv2.rectangle(image_end, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 1)

        preview_text = "".join(preview)
        print(preview_text)

        #return preview_text
        #file_name = os.path.basename(file)
        #cv2.imwrite(f"identified/{file_name}", image_end)

        print("Texto de visualização original: ", preview_text)

        # Aplicar regras apenas se o preview_text tiver mais de 5 caracteres
        if len(preview_text) == 6:
            # Aplicar regras para sequências
            sequencias_para_substituir = {"ni": "m",
                                          "nl": "m",
                                          "in": "m",
                                          "ln": "m",
                                          "fi": "n",
                                          "fl": "n"}  # Adicione mais conforme necessário

            for seq, substituicao in sequencias_para_substituir.items():
                preview_text = preview_text.replace(seq, substituicao)

        if len(preview_text) == 7:
            # Aplicar regras para sequências
            sequencias_para_substituir = {"nni": "m",
                                          "nil": "m",
                                          "lii": "m",
                                          "nln": "m",
                                          "nll": "m",
                                          "nli": "m",
                                          "nin": "m",
                                          "nil": "m",
                                          "nii": "m",
                                          "lnn": "m",
                                          "lnl": "m",
                                          "lni": "m",
                                          "lln": "m",
                                          "lll": "m",
                                          "lli": "m",
                                          "lin": "m",
                                          "lil": "m",
                                          "lii": "m",
                                          "inn": "m",
                                          "inl": "m",
                                          "ini": "m",
                                          "iln": "m",
                                          "ill": "m",
                                          "ili": "m",
                                          "iin": "m",
                                          "iil": "m",
                                          "iii": "m"}  # Adicione mais conforme necessário

            for seq, substituicao in sequencias_para_substituir.items():
                preview_text = preview_text.replace(seq, substituicao)
            # Truncar para 5 caracteres se for mais longo
            # preview_text = preview_text[:5]

        print("Texto de visualização processado:", preview_text)

    print("Processamento concluído.")
##################

if __name__ == "__main__":
    solve_captcha()