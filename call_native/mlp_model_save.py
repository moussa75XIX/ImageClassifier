import os
import ctypes
import numpy as np
from PIL import Image

MAIN_DATASET_FOLDER = "..\\dataset\\"

TEST_SUBFOLDER = os.path.join(MAIN_DATASET_FOLDER, "test")
TRAIN_SUBFOLDER = os.path.join(MAIN_DATASET_FOLDER, "train")

TEST_ESPAGNE_FOLDER = os.path.join(TEST_SUBFOLDER, "espagne")
TEST_FRANCE_FOLDER = os.path.join(TEST_SUBFOLDER, "france")
TEST_JAPON_FOLDER = os.path.join(TEST_SUBFOLDER, "japon")

TRAIN_ESPAGNE_FOLDER = os.path.join(TRAIN_SUBFOLDER, "espagne")
TRAIN_FRANCE_FOLDER = os.path.join(TRAIN_SUBFOLDER, "france")
TRAIN_JAPON_FOLDER = os.path.join(TRAIN_SUBFOLDER, "japon")

PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary"
MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)

def fill_x_and_y_with_images_and_labels(folder, classe, dataset):

    for file in os.listdir(folder):

        file_path = os.path.join(folder, file)
        image = Image.open(file_path)
        image = image.resize((8, 8))

        im_arr = np.array(image).flatten()
        im_arr = im_arr / 255.0

        completed_dataset = {}
        completed_dataset["value"] = im_arr
        completed_dataset["classe"] = classe

        dataset.append(completed_dataset.copy())


def import_dataset(folder, classe):

    dataset = []

    fill_x_and_y_with_images_and_labels(folder, classe, dataset)

    return dataset


def run():

    x_test = []
    y_test = []

    x_train = []
    y_train = []

    class_espagne_expected_outputs = [1, -1, -1]
    dataset_test_for_espagne = import_dataset(TEST_ESPAGNE_FOLDER, class_espagne_expected_outputs)
    dataset_train_for_espagne = import_dataset(TRAIN_ESPAGNE_FOLDER, class_espagne_expected_outputs)

    class_france_expected_outputs = [-1, 1, -1]
    dataset_test_for_france = import_dataset(TEST_FRANCE_FOLDER, class_france_expected_outputs)
    dataset_train_for_france = import_dataset(TRAIN_FRANCE_FOLDER, class_france_expected_outputs)

    class_japon_expected_outputs = [-1, -1, 1]
    dataset_test_for_japon = import_dataset(TEST_JAPON_FOLDER, class_japon_expected_outputs)
    dataset_train_for_japon = import_dataset(TRAIN_JAPON_FOLDER, class_japon_expected_outputs)

    final_dataset_test = dataset_test_for_espagne + dataset_test_for_france + dataset_test_for_japon
    final_dataset_train = dataset_train_for_espagne + dataset_train_for_france + dataset_train_for_japon

    for value in final_dataset_test:

        the_image = value["value"].tolist()
        the_classe = value["classe"]

        x_test.append(the_image)
        y_test.append(the_classe)

    for value in final_dataset_train:

        the_image = value["value"].tolist()
        the_classe = value["classe"]

        x_train.append(the_image)
        y_train.append(the_classe)

    dataset_inputs = x_train
    dataset_outputs = y_train

    initiate_dataset_inputs_size = len(dataset_inputs[0])
    initiate_dataset_inputs_type = ctypes.c_float * initiate_dataset_inputs_size

    classes_initiate_dataset_inputs_size = initiate_dataset_inputs_size * 2

    inputs_result = []

    for sublist in dataset_inputs:
        for item in sublist:
            inputs_result.append(item)

    dataset_inputs = inputs_result

    outputs_result = []

    for sublist in dataset_outputs:
        for item in sublist:
            outputs_result.append(item)

    dataset_outputs = outputs_result

    arr = [initiate_dataset_inputs_size, classes_initiate_dataset_inputs_size, 3]
    arr_size = len(arr)
    arr_type = ctypes.c_int * arr_size

    # Create model
    MY_LIB.create_mlp_model.argtypes = [arr_type, ctypes.c_int]
    MY_LIB.create_mlp_model.restype = ctypes.c_void_p

    native_arr = arr_type(*arr)

    model = MY_LIB.create_mlp_model(native_arr, arr_size)

    dataset_inputs_size = len(dataset_inputs)
    dataset_outputs_size = len(dataset_outputs)

    dataset_inputs_type = ctypes.c_float * dataset_inputs_size
    dataset_outputs_type = ctypes.c_float * dataset_outputs_size

    MY_LIB.train_classification_stochastic_backprop_mlp_model.argtypes = [ctypes.c_void_p, dataset_inputs_type,
                                                                          ctypes.c_int,
                                                                          dataset_outputs_type,
                                                                          ctypes.c_int, ctypes.c_float, ctypes.c_int]

    MY_LIB.train_classification_stochastic_backprop_mlp_model.restype = None

    native_dataset_inputs = dataset_inputs_type(*dataset_inputs)
    native_dataset_outputs = dataset_outputs_type(*dataset_outputs)
    
    MY_LIB.train_classification_stochastic_backprop_mlp_model(model, native_dataset_inputs, dataset_inputs_size,
                                                              native_dataset_outputs, dataset_outputs_size, 0.001,
                                                              100000)

    predicted_values = []


    for p in x_train:

        MY_LIB.predict_mlp_model_classification.argtypes = [ctypes.c_void_p, initiate_dataset_inputs_type, ctypes.c_int]
        MY_LIB.predict_mlp_model_classification.restype = ctypes.POINTER(ctypes.c_float)

        native_p = initiate_dataset_inputs_type(*p)

        predict = MY_LIB.predict_mlp_model_classification(model, native_p, initiate_dataset_inputs_size)

        tab_predict = []

        for q in range(3):
            tab_predict.append(predict[q])

        predicted_values.append(tab_predict)

    path = "..\\saves\\mlp_model\\"
    b_path = path.encode('utf-8')

    MY_LIB.save_mlp_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    MY_LIB.save_mlp_model.restype = None
    MY_LIB.save_mlp_model(model, b_path)  # Model save

    MY_LIB.destroy_mlp_model.argtypes = [ctypes.c_void_p]
    MY_LIB.destroy_mlp_model.restype = None
    MY_LIB.destroy_mlp_model(model)  # Model erase


if __name__ == "__main__":
    run()
