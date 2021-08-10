//
// Created by houci on 17/07/2021.
//

#include "LINEAR.h"
#include "json.hpp"
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

using json = nlohmann::json;

DLLEXPORT float* create_linear_model(int initiate_size) {

    float* model = new float[initiate_size];

    std::mt19937_64 rng;

    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count(); // initialize the random number generator with time-dependent seed
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);

    std::uniform_real_distribution<float> unif(0.0, 1.0); // initialize a uniform distribution between 0 and 1

    for(int idx = 0; idx < initiate_size; idx++) {

        float random_value = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * 2 - 1;
        model[idx] = random_value;

    }

    return model;

}

DLLEXPORT float predict_linear_model_regression(float* model, int model_size, float* inputs) {

    float sum_rslt = model[0];

    for (int i = 1; i < model_size; i++)
        sum_rslt += model[i] * inputs[i - 1];

    return sum_rslt;

}

DLLEXPORT float predict_linear_model_classification(float* model, int model_size, float* inputs) {

    float pred = predict_linear_model_regression(model, model_size, inputs);

    if (pred >= 0)
        return 1.0;

    else
        return -1.0;

}

DLLEXPORT void train_rosenblatt_linear_model(float* model, int model_size,
                                             float* dataset_inputs, int dataset_inputs_size,
                                             float* dataset_expected_outputs, int outputs_size,
                                             int iterations_count = 10,
                                             float alpha = 0.01) {

    int input_size = model_size - 1;
    int sample_count = floor(dataset_inputs_size / input_size);

    for (int it = 0; it < iterations_count; it++) {

        int k = rand() % (sample_count);

        int Xk_size = ((k+1) * input_size) - (k * input_size);
        float Xk[Xk_size];

        for (int iteration = k * input_size, x = 0; iteration < (k + 1) * input_size && x < Xk_size; iteration++, x++)
                Xk[x] = dataset_inputs[iteration];

        int yk = dataset_expected_outputs[k * 1];

        float gXk = predict_linear_model_classification(model, model_size, Xk);

        model[0] += alpha * (yk - gXk) * 1.0;

        for (int i = 1; i < model_size; i++)
            model[i] += alpha * (yk - gXk) * Xk[i - 1];

    }

}

/*DLLEXPORT void train_regression_linear_model(float* tab_model, int model_size, float* tab_dataset_inputs, int dataset_input_size, float* tab_dataset_expected_outputs, int dataset_output_size) {

    std::vector<float> model(tab_model, tab_model + model_size);
    std::vector<float> dataset_inputs(tab_dataset_inputs, tab_dataset_inputs + dataset_input_size);
    std::vector<float> dataset_expected_outputs (tab_dataset_expected_outputs, tab_dataset_expected_outputs + dataset_output_size);

    int input_size = model.size() - 1;
    int sample_count = floor(dataset_inputs.size() / input_size);

    std::vector<float> X = dataset_inputs;
    std::vector<float> Y = dataset_expected_outputs;

    nc::NdArray<float> arr_X = X;
    nc::NdArray<float> arr_Y = Y;

    arr_X.reshape(sample_count, input_size);

    std::cout << arr_X[0];

    for(auto&a : arr_X)
        std::cout << "\nval resh : " << a << " ";
    std::cout << std::endl;
    //np.reshape

    std::vector<float> samples (sample_count);
    nc::NdArray<float> bias_fake_inputs = samples;
    bias_fake_inputs.ones();

    for(auto&a : bias_fake_inputs)
        std::cout << "\nval ones : " << a << " ";
    std::cout << std::endl;

    arr_Y.reshape(sample_count, 1);

    for(auto&a : arr_Y)
        std::cout << "\nY resh : " << a << " ";
    std::cout << std::endl;

    //np.hstack


    //np.matmul with multiplyDLLEXPORT void train_regression_linear_model(LINEAR* model, float* tab_dataset_inputs, int input_size, float* tab_dataset_expected_outputs, int output_size) {

    //transpose

    //for(int i = 0; i < model_size; i++)
        //model[i] = W[i][0]

}*/

DLLEXPORT void save_linear_model(float *model, int size, const char* classe, const char* tab_path) {

    std::string sclasse(classe);
    std::string deb_path(tab_path);

    std::string beg = "train_linear_model_" + sclasse + "_";

    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%d_%m_%Y_H%H_M%M_S%S", &tstruct);

    std::string filename = beg + buf;
    std::string path = deb_path + "\\" + filename + ".json";

    std::ofstream file;
    file.open(path);

    std::vector results_model(model, model + size);

    json object;

    std::string linear_class = "linear_model";

    object[linear_class] = results_model;

    file << std::setw(4) << object << std::endl;

    file.close();

}

DLLEXPORT float* load_linear_model(const char* filename) {

    std::string str_file(filename);

    std::string path(str_file);

    std::ifstream  file(path);

    json object;
    file >> object;

    json model_values = object["linear_model"];

    std::vector<float> vec_model = model_values;

    float* model = new float[vec_model.size()];

    for(int value = 0; value < vec_model.size(); value++)
        model[value] = vec_model[value];

    return model;

}

DLLEXPORT void destroy_linear_model(float* model) {

    delete[](model);

}