import os
import keras.utils as image
from keras.models import load_model
import pandas as pd
from data_loader import preprocess
import numpy as np

model = load_model("./model/_mini_XCEPTION.h5")
# model.load_weights('./model/face_classify.h5')

face = ''
w = open("./result/image_pre.csv", mode='a')
w.write("ID,Category\n")

os.chdir('./test')
img_d = os.listdir()
# img_d.remove("result")
key = lambda i : int(i.split('.')[0])
new_img = sorted(img_d, key=key)

for file_name in new_img:
    img = image.load_img(file_name,target_size=(48,48),color_mode='grayscale')
    x = image.img_to_array(img)
    x = preprocess(x)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    num = np.argmax(classes)

    answer = file_name.split('.')[0] + ',' + str(num) + '\n'
    w.write(answer)

w.close()