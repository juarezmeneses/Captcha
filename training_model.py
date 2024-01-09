import glob

import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.layers import Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from helpers import resize_to_fit
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


import numpy as np


print("Versão do NumPy:", np.__version__)

data  = []
labels = []
folder_base_images = "bd_letters_and_numbers_labeled"

#images = paths.list_images(folder_base_images)
images = glob.glob(os.path.join(folder_base_images, "*", "*.png"))
print(list(images))

image_count = 0

for file in images:
    image_count += 1
    print("Entrou no for: ")
    # ['bd_letters_and_numbers_labeled', 'A_upper', "image.png"]
    label = file.split(os.path.sep)[-2]
    print("Label: ", label)
    image = cv2.imread(file)
    if image is None:
        raise Exception(f"Erro ao ler a imagem: {file}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Dimensões antes do resize:", image.shape)

    image = resize_to_fit(image, 20, 20)   #[Value, Value, x]
    print("Dimensões após o resize:", image.shape)

    image = np.expand_dims(image, axis=2)

    labels.append(label)
    data.append(image)

print("Número total de imagens encontradas:", image_count)

print(list(data))
print(list(labels))

print("Número total de dados:", len(data))
print("Número total de rótulos:", len(labels))

data = np.array(data, dtype="float") / 255
labels = np.array(labels)

print("Número total de dados2:", len(data))
print("Número total de rótulos2:", len(labels))

#training(75%) and test(25%)
#X=data and Y=label
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

print("Número total de dados:", len(data))
print("Número de dados de treinamento:", len(X_train))
print("Número de dados de teste:", len(X_test))

if len(X_train) == 0:
    print("Erro: O conjunto de treinamento está vazio. Verifique seus dados.")
    exit()

#matrix of labels
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

with open('labels_model.dat', 'wb') as file_pickle:
    pickle.dump(lb, file_pickle)

#ia
model = Sequential()

#layers
#1
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#2
model.add(Conv2D(50, (5, 5), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#3
model.add(Flatten())
model.add(Dense(500, activation="relu"))

#4 out
model.add(Dense(58, activation="softmax"))

#compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#training
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=58, epochs=10, verbose=1)

#save model
model.save("trained_model.hdf5")