from keras import layers, models  # condaでkerasをインストール
from keras import optimizers
from keras.utils import np_utils
from PIL import Image  # condaでpillowをインストール
import os
import glob
import numpy as np
import random
import math
import matplotlib.pyplot as plt

root_dir = "~/python/tea"
categories = ["apple", "banana", "orange"]

X = []
Y = []


def make_sample(files):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X), np.array(Y)


def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    # ２次元を１次元に変換していると思う。多分、asarrayにすることでメモリを共有するから、arrayより早いのか？
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)


allfiles = []

for idx, cat in enumerate(categories):
    image_dir = cat
    # qiitaでは絶対パスだが、俺は相対パスでないとファイルが取得されない。
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))

random.shuffle(allfiles)
# print("allfilesの数：" + str(len(allfiles)))
th = math.floor(len(allfiles) * 0.8)
train = allfiles[0:th]
test = allfiles[th:]
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)
xy = (X_train, X_test, y_train, y_test)
# np.save("xy_data/xy.npy", xy) # 保存してもいつ使うんだ？


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(3, activation="sigmoid"))

model.summary()

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])


X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

nb_classes = len(categories)
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = model.fit(X_train, y_train, epochs=10, batch_size=6,
                  validation_data=(X_test, y_test))
