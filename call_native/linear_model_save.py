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
        image = image.resize((64, 64))

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

    initiate_dataset_inputs_size = len(dataset_inputs[0])
    initiate_dataset_inputs_type = ctypes.c_float * initiate_dataset_inputs_size

    result = []

    for sublist in dataset_inputs:
        for item in sublist:
            result.append(item)

    dataset_inputs = result

    y_train = np.array(y_train)

    tmp_dataset_expected_outputs = []

    espagne_dataset_expected_outputs = []
    france_dataset_expected_outputs = []
    japon_dataset_expected_outputs = []

    # espagne & tmp expected outputs
    for espagne_arr in y_train:
        espagne_dataset_expected_outputs.append(espagne_arr[0])
        tmp_dataset_expected_outputs.append(espagne_arr[0])

    # france expected outputs
    for france_arr in y_train:
        france_dataset_expected_outputs.append(france_arr[1])

    # japon expected outputs
    for japon_arr in y_train:
        japon_dataset_expected_outputs.append(japon_arr[2])

    dim = initiate_dataset_inputs_size

    MY_LIB.create_linear_model.argtypes = [ctypes.c_int]
    MY_LIB.create_linear_model.restype = ctypes.POINTER(ctypes.c_float)

    temp_model = MY_LIB.create_linear_model(dim)
    temp_model_type = type(temp_model)

    espagne_model = MY_LIB.create_linear_model(dim)
    france_model = MY_LIB.create_linear_model(dim)
    japon_model = MY_LIB.create_linear_model(dim)

    final_size = initiate_dataset_inputs_size + 1

    dataset_inputs_size = len(dataset_inputs)
    dataset_outputs_size = len(tmp_dataset_expected_outputs)

    dataset_inputs_type = ctypes.c_float * dataset_inputs_size
    dataset_outputs_type = ctypes.c_float * dataset_outputs_size

    MY_LIB.train_rosenblatt_linear_model.argtypes = [temp_model_type, ctypes.c_int, dataset_inputs_type,
                                                     ctypes.c_int, dataset_outputs_type, ctypes.c_int, ctypes.c_int,
                                                     ctypes.c_float]

    MY_LIB.train_rosenblatt_linear_model.restype = None

    espagne_dataset_outputs_size = len(espagne_dataset_expected_outputs)
    france_dataset_outputs_size = len(france_dataset_expected_outputs)
    japon_dataset_outputs_size = len(japon_dataset_expected_outputs)

    espagne_dataset_outputs_type = ctypes.c_float * espagne_dataset_outputs_size
    france_dataset_outputs_type = ctypes.c_float * france_dataset_outputs_size
    japon_dataset_outputs_type = ctypes.c_float * japon_dataset_outputs_size

    native_dataset_inputs = dataset_inputs_type(*dataset_inputs)

    native_espagne_dataset_outputs = espagne_dataset_outputs_type(*espagne_dataset_expected_outputs)
    native_france_dataset_outputs = france_dataset_outputs_type(*france_dataset_expected_outputs)
    native_japon_dataset_outputs = japon_dataset_outputs_type(*japon_dataset_expected_outputs)

    MY_LIB.train_rosenblatt_linear_model(espagne_model, final_size, native_dataset_inputs,
                                         dataset_inputs_size, native_espagne_dataset_outputs,
                                         espagne_dataset_outputs_size, 100000, 0.001)

    MY_LIB.train_rosenblatt_linear_model(france_model, final_size, native_dataset_inputs,
                                         dataset_inputs_size, native_france_dataset_outputs,
                                         france_dataset_outputs_size, 100000, 0.001)

    MY_LIB.train_rosenblatt_linear_model(japon_model, final_size, native_dataset_inputs,
                                         dataset_inputs_size, native_japon_dataset_outputs,
                                         japon_dataset_outputs_size, 100000, 0.001)

    predicted_values = []

    for arr in x_train:

        tab_predicts = []

        MY_LIB.predict_linear_model_classification.argtypes = [temp_model_type, ctypes.c_int,
                                                               initiate_dataset_inputs_type,
                                                               ctypes.c_int]

        MY_LIB.predict_linear_model_classification.restype = ctypes.c_float

        native_arr_type = initiate_dataset_inputs_type(*arr)

        espagne_predict = MY_LIB.predict_linear_model_classification(espagne_model, initiate_dataset_inputs_size,
                                                                    native_arr_type, final_size)

        france_predict = MY_LIB.predict_linear_model_classification(france_model, initiate_dataset_inputs_size,
                                                                    native_arr_type, final_size)

        japon_predict = MY_LIB.predict_linear_model_classification(japon_model, initiate_dataset_inputs_size,
                                                                    native_arr_type, final_size)

        tab_predicts.append(espagne_predict)
        tab_predicts.append(france_predict)
        tab_predicts.append(japon_predict)

        predicted_values.append(tab_predicts.copy())

    path = "..\\saves\\linear_model\\"
    b_path = path.encode('utf-8')

    class_espagne = "espagne"
    class_france = "france"
    class_japon = "japon"

    b_class_espagne = class_espagne.encode('utf-8')
    b_class_france = class_france.encode('utf-8')
    b_class_japon = class_japon.encode('utf-8')

    MY_LIB.save_linear_model.argtypes = [type(temp_model), ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
    MY_LIB.save_linear_model.restype = None

    # Model save
    MY_LIB.save_linear_model(espagne_model, final_size, b_class_espagne, b_path)
    MY_LIB.save_linear_model(france_model, final_size, b_class_france, b_path)
    MY_LIB.save_linear_model(japon_model, final_size, b_class_japon, b_path)

    MY_LIB.destroy_linear_model.argtypes = [ctypes.c_void_p]
    MY_LIB.destroy_linear_model.restype = None

    # Model erase
    MY_LIB.destroy_linear_model(espagne_model)
    MY_LIB.destroy_linear_model(france_model)
    MY_LIB.destroy_linear_model(japon_model)


if __name__ == "__main__":
    run()
