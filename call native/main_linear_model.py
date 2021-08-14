import os
import ctypes
import numpy as np
from PIL import Image

PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary"
MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)

CLASSES = ["espagne", "france", "japon"]
CLASSES_SIZE = len(CLASSES)

def get_classe(folder, file, espagne_predict, france_predict, japon_predict):

    file_path = os.path.join(folder, file)
    image = Image.open(file_path)
    image = image.resize((64, 64))
    im_arr = np.array(image).flatten()
    im_arr = im_arr / 255.0
    im_arr = im_arr.tolist()

    MY_LIB.load_linear_model.argtypes = [ctypes.c_char_p]
    MY_LIB.load_linear_model.restype = ctypes.POINTER(ctypes.c_float)

    espagne_b_path = espagne_predict.encode('utf-8')
    france_b_path = france_predict.encode('utf-8')
    japon_b_path = japon_predict.encode('utf-8')

    espagne_model = MY_LIB.load_linear_model(espagne_b_path)
    france_model = MY_LIB.load_linear_model(france_b_path)
    japon_model = MY_LIB.load_linear_model(japon_b_path)

    dataset_inputs_size = len(im_arr)
    dataset_inputs_type = ctypes.c_float * dataset_inputs_size

    MY_LIB.predict_linear_model_classification.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, dataset_inputs_type]
    MY_LIB.predict_linear_model_classification.restype = ctypes.c_float

    native_dataset_inputs = dataset_inputs_type(*im_arr)

    espagne_predict = MY_LIB.predict_linear_model_classification(espagne_model, dataset_inputs_size, native_dataset_inputs)
    france_predict = MY_LIB.predict_linear_model_classification(france_model, dataset_inputs_size, native_dataset_inputs)
    japon_predict = MY_LIB.predict_linear_model_classification(japon_model, dataset_inputs_size, native_dataset_inputs)

    tab_predict = []
    tab_predict.append(espagne_predict)
    tab_predict.append(france_predict)
    tab_predict.append(japon_predict)
    print(tab_predict)

    final_prediction = []

    for i in range(CLASSES_SIZE):

        if tab_predict[i] == 1:
            final_prediction.append(CLASSES[i])

    return final_prediction


folder = "..\\dataset\\test\\japon"
file = "563.png"

espagne_file_model = "..\\saves\\linear_model\\train_linear_model_espagne_14_08_2021_H17_M51_S58.json"
france_file_model = "..\\saves\\linear_model\\train_linear_model_france_14_08_2021_H17_M51_S58.json"
japon_file_model = "..\\saves\\linear_model\\train_linear_model_japon_14_08_2021_H17_M51_S58.json"

prediction = get_classe(folder, file, espagne_file_model, france_file_model, japon_file_model)
print(prediction)