import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os

PATH_TO_SHARED_LIBRARY = "../library/cmake-build-debug/liblibrary"
MY_LIB = ctypes.CDLL(PATH_TO_SHARED_LIBRARY)

if __name__ == "__main__":
    d = [2, 2, 1]
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    Y = np.array([1.0, 1.0, -1.0, -1.0])

    X_test = [2.0, 3.0]

    X_flat = []
    for elt in X:
        X_flat.append(elt[0])
        X_flat.append(elt[1])

    arr_size = len(d)
    arr_type = ctypes.c_int * arr_size
    native_arr = arr_type(*d)
    MY_LIB.create_mlp_model.argtypes = [arr_type, ctypes.c_int]
    MY_LIB.create_mlp_model.restype = ctypes.c_void_p

    mlp = MY_LIB.create_mlp_model(native_arr, arr_size)

    x_test_size = len(X_test)
    x_test_type = ctypes.c_float * x_test_size
    MY_LIB.predict_mlp_model_classification.argtypes = [ctypes.c_void_p, x_test_type, ctypes.c_int]
    MY_LIB.predict_mlp_model_classification.restype = ctypes.POINTER(ctypes.c_float)
    x_test_native = x_test_type(*X_test)
    pred = MY_LIB.predict_mlp_model_classification(mlp, x_test_native, x_test_size)
    np_arr = np.ctypeslib.as_array(pred, (1,))
    print(np_arr)

    x_size = len(X_flat)
    x_type = ctypes.c_float * x_size
    y_size = len(Y)
    y_type = ctypes.c_float * y_size
    MY_LIB.train_classification_stochastic_backprop_mlp_model.argtypes = [ctypes.c_void_p, x_type, y_type, ctypes.c_float, ctypes.c_int,
                                                                          ctypes.c_int, ctypes.c_int]
    MY_LIB.train_classification_stochastic_backprop_mlp_model.restype = None
    x_native = x_type(*X_flat)
    y_native = y_type(*Y)
    MY_LIB.train_classification_stochastic_backprop_mlp_model(mlp, x_native, y_native, 0.01, 100, x_size, y_size)

    x_test_size = len(X_test)
    x_test_type = ctypes.c_float * x_test_size
    MY_LIB.predict_mlp_model_classification.argtypes = [ctypes.c_void_p, x_test_type, ctypes.c_int]
    MY_LIB.predict_mlp_model_classification.restype = ctypes.POINTER(ctypes.c_float)
    x_test_native = x_test_type(*X_test)
    pred = MY_LIB.predict_mlp_model_classification(mlp, x_test_native, x_test_size)
    np_arr = np.ctypeslib.as_array(pred, (1,))
    print(np_arr)

    MY_LIB.destroy_model.argtypes = [ctypes.c_void_p]
    MY_LIB.destroy_model.restype = None
    MY_LIB.destroy_model(mlp)