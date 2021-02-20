from PIL import Image
import os
import glob
import numpy as np
import random
import math

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
print("allfilesの数：" + str(len(allfiles)))
th = math.floor(len(allfiles) * 0.8)
train = allfiles[0:th]
test = allfiles[th:]
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)
xy = (X_train, X_test, y_train, y_test)
np.save("xy_data/xy.npy", xy)
