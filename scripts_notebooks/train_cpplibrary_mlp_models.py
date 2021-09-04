#!/usr/bin/env python
# coding: utf-8

import os
import ctypes
import numpy as np
from PIL import Image


PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary"
MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)


MAIN_DATASET_FOLDER = "..\\dataset\\"

TEST_SUBFOLDER = os.path.join(MAIN_DATASET_FOLDER, "test")
TRAIN_SUBFOLDER = os.path.join(MAIN_DATASET_FOLDER, "train")

TEST_ESPAGNE_FOLDER = os.path.join(TEST_SUBFOLDER, "espagne")
TEST_FRANCE_FOLDER = os.path.join(TEST_SUBFOLDER, "france")
TEST_JAPON_FOLDER = os.path.join(TEST_SUBFOLDER, "japon")

TRAIN_ESPAGNE_FOLDER = os.path.join(TRAIN_SUBFOLDER, "espagne")
TRAIN_FRANCE_FOLDER = os.path.join(TRAIN_SUBFOLDER, "france")
TRAIN_JAPON_FOLDER = os.path.join(TRAIN_SUBFOLDER, "japon")


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


x_test = []
y_test = []

x_train = []
y_train = []

class_espagne_expected_outputs = [1, 0, 0]
dataset_test_for_espagne = import_dataset(TEST_ESPAGNE_FOLDER, class_espagne_expected_outputs)
dataset_train_for_espagne = import_dataset(TRAIN_ESPAGNE_FOLDER, class_espagne_expected_outputs)

class_france_expected_outputs = [0, 1, 0]
dataset_test_for_france = import_dataset(TEST_FRANCE_FOLDER, class_france_expected_outputs)
dataset_train_for_france = import_dataset(TRAIN_FRANCE_FOLDER, class_france_expected_outputs)

class_japon_expected_outputs = [0, 0, 1]
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

# Load model_a
MY_LIB.load_mlp_model.argtypes = [ctypes.c_char_p]
MY_LIB.load_mlp_model.restype = ctypes.c_void_p

filepath = "..\\models\\cpplibrary_models\\model_a.json".encode('utf-8')
model_a = MY_LIB.load_mlp_model(filepath)

# Load model_b
MY_LIB.load_mlp_model.argtypes = [ctypes.c_char_p]
MY_LIB.load_mlp_model.restype = ctypes.c_void_p

filepath = "..\\models\\cpplibrary_models\\model_b.json".encode('utf-8')
model_b = MY_LIB.load_mlp_model(filepath)

# Load model_c
MY_LIB.load_mlp_model.argtypes = [ctypes.c_char_p]
MY_LIB.load_mlp_model.restype = ctypes.c_void_p

filepath = "..\\models\\cpplibrary_models\\model_c.json".encode('utf-8')
model_c = MY_LIB.load_mlp_model(filepath)

# Load model_d
MY_LIB.load_mlp_model.argtypes = [ctypes.c_char_p]
MY_LIB.load_mlp_model.restype = ctypes.c_void_p

filepath = "..\\models\\cpplibrary_models\\model_d.json".encode('utf-8')
model_d = MY_LIB.load_mlp_model(filepath)


def train(model, name):

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

    path = "..\\models\\cpplibrary_models\\".encode('utf-8')

    name = name.encode('utf-8')

    # Save model
    MY_LIB.save_mlp_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    MY_LIB.save_mlp_model.restype = None
    MY_LIB.save_mlp_model(model, path, name)


train(model_a, "a")
#train(model_b, "b")
#train(model_c, "c")
#train(model_d, "d")




