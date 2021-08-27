import os
import ctypes
import numpy as np
from PIL import Image

PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary"
MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)

CLASSES = ["enfant", "adulte", "senior"]
CLASSES_SIZE = len(CLASSES)

folder = "..\\dataset\\test\\adulte"
file = "2995.png"

full_path = folder + "\\" + file

file_path = os.path.join(folder, file)

image = Image.open(file_path)
if image.mode in ("RGB", "P"): image = image.convert("RGBA")

image = image.resize((8, 8))

im_arr = np.array(image).flatten()
im_arr = im_arr / 255.0
im_arr = im_arr.tolist()

MY_LIB.load_mlp_model.argtypes = [ctypes.c_char_p]
MY_LIB.load_mlp_model.restype = ctypes.c_void_p

file_model = "..\\saves\\mlp_model\\train_mlp_model_27_07_2021_H17_M58_S48.json"
b_path = file_model.encode('utf-8')

model = MY_LIB.load_mlp_model(b_path)

dataset_inputs_size = len(im_arr)
dataset_inputs_type = ctypes.c_float * dataset_inputs_size

MY_LIB.predict_mlp_model_classification.argtypes = [ctypes.c_void_p, dataset_inputs_type, ctypes.c_int]
MY_LIB.predict_mlp_model_classification.restype = ctypes.POINTER(ctypes.c_float)

native_dataset_inputs = dataset_inputs_type(*im_arr)

predict = MY_LIB.predict_mlp_model_classification(model, native_dataset_inputs, dataset_inputs_size)

tab_predict = []

for q in range(3):
    tab_predict.append(predict[q])

print(tab_predict)

final_prediction = []

for i in range(CLASSES_SIZE):

    if tab_predict[i] > 0.7:
        final_prediction.append(CLASSES[i])

if tab_predict[1] < 0.7 and tab_predict[2] < 0.7:
    final_prediction.append(CLASSES[0])

print(final_prediction)