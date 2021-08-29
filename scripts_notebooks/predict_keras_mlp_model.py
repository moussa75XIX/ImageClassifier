import tensorflow as tf
import numpy as np
import ctypes
from PIL import Image


model = tf.keras.models.load_model("../models/keras_models/model_d.h5")
CLASSES = ["espagne", "france", "japon"]
CLASSES_SIZE = len(CLASSES)

img_path = r"..\dataset\test\france\267.png"


def predict(model,CLASSES,CLASSES_SIZE,img_path):

    image = Image.open(img_path)
    image = image.resize((64, 64))
    if image.mode in ("RGB", "P"): image = image.convert("RGBA")

    im_arr = np.array(image).flatten()
    im_arr = im_arr / 255.0
    im_arr = im_arr.tolist()

    dataset_inputs_size = len(im_arr)
    dataset_inputs_type = ctypes.c_float * dataset_inputs_size

    im_arr = np.array(im_arr)

    img_batch = np.expand_dims(im_arr, axis=0)

    prediction = model.predict(img_batch)

    prediction_list = prediction.tolist()

    flat_list = []
    for sublist in prediction_list:
        for item in sublist:
            flat_list.append(item)

            tab_predict = []

    for q in range(3):
        tab_predict.append(flat_list[q])

    max_tab = max(tab_predict)
    prediction_percent = 0

    for i in range(CLASSES_SIZE):
        if tab_predict[i] == max_tab:
            prediction_percent = (tab_predict[i])

    if prediction_percent == tab_predict[0]:
        prediction_class = CLASSES[0]
    elif prediction_percent == tab_predict[1]:
        prediction_class = CLASSES[1]
    else:
        prediction_class = CLASSES[2]

    print(tab_predict)
    print(prediction_class)

    return prediction_class


def main():
    predict(model, CLASSES, CLASSES_SIZE, img_path)


if __name__ == "__main__":
    main()