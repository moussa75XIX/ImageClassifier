import ctypes
import numpy as np
from PIL import Image

PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary"
MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)

CLASSES = ["espagne", "france", "japon"]
CLASSES_SIZE = len(CLASSES)
img_path = r"..\dataset\test\france\283.png"

MY_LIB.load_mlp_model.argtypes = [ctypes.c_char_p]
MY_LIB.load_mlp_model.restype = ctypes.c_void_p

filepath = "..\\models\\cpplibrary_models\\model_a.json"
filepath = filepath.encode('utf-8')
model = MY_LIB.load_mlp_model(filepath)


def predict(model,CLASSES,CLASSES_SIZE,img_path):
    image = Image.open(img_path)
    if image.mode in ("RGB", "P"): image = image.convert("RGBA")

    image = image.resize((64, 64))

    im_arr = np.array(image).flatten()
    im_arr = im_arr / 255.0
    im_arr = im_arr.tolist()

    MY_LIB.load_mlp_model.argtypes = [ctypes.c_char_p]
    MY_LIB.load_mlp_model.restype = ctypes.c_void_p

    dataset_inputs_size = len(im_arr)
    dataset_inputs_type = ctypes.c_float * dataset_inputs_size

    MY_LIB.predict_mlp_model_classification.argtypes = [ctypes.c_void_p, dataset_inputs_type, ctypes.c_int]
    MY_LIB.predict_mlp_model_classification.restype = ctypes.POINTER(ctypes.c_float)

    native_dataset_inputs = dataset_inputs_type(*im_arr)

    predict = MY_LIB.predict_mlp_model_classification(model, native_dataset_inputs, dataset_inputs_size)

    tab_predict = []

    for q in range(3):
        tab_predict.append(predict[q])

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
