#include "MLP.h"
#include "json.hpp"
#include <stdlib.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT extern "C"
#endif

using json = nlohmann::json;

class MLP {
public:
    std::vector<int> d;
    std::vector<std::vector<std::vector<float>>> W;
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> deltas;

    MLP(std::vector<int> _d,
        std::vector<std::vector<std::vector<float>>> _W,
        std::vector<std::vector<float>> _X,
        std::vector<std::vector<float>> _deltas) {

        d = _d;
        W = _W;
        X = _X;
        deltas = _deltas;

    }

    void forward_pass(float* tab_sample_inputs, int size, bool is_classification) {

        std::vector<float> sample_inputs(tab_sample_inputs, tab_sample_inputs + size);

        for (int j = 1; j < this->d[0] + 1; j++)
            this->X[0][j] = sample_inputs[j - 1];

        for (int l = 1; l < this->d.size(); l++) {

            for (int j = 1; j < this->d[l] + 1; j++) {

                float sum_result = 0.0;

                for (int i = 0; i < this->d[l - 1] + 1; i++)
                    sum_result += this->W[l][i][j] * this->X[l - 1][i];

                this->X[l][j] = sum_result;

                if ((l < this->d.size() - 1) || (is_classification == true))
                    this->X[l][j] = tanh(this->X[l][j]);

            }

        }

    }

    void train_stochastic_gradient_backpropagation(float* tab_flattened_dataset_inputs,
                                                   int input_size,
                                                   float* tab_flattened_expected_outputs,
                                                   int output_size,
                                                   bool is_classification,
                                                   float alpha = 0.01,
                                                   int iterations_count = 1000) {

        std::vector<float> flattened_dataset_inputs(tab_flattened_dataset_inputs, tab_flattened_dataset_inputs + input_size);
        std::vector<float> flattened_expected_outputs(tab_flattened_expected_outputs, tab_flattened_expected_outputs + output_size);

        int L = this->d.size() - 1;
        float input_dim = this->d[0];
        float output_dim = this->d[L];
        int samples_count = floor(flattened_dataset_inputs.size() / input_dim);

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(0,(samples_count - 1));

        for (int it = 0; it < iterations_count; it++) {

            int k = distribution(generator);

            std::vector<float> sample_inputs;

            for (int iter = k * input_dim; iter < (k + 1) * input_dim; iter++)
                sample_inputs.push_back(flattened_dataset_inputs[iter]);

            std::vector<float> sample_expected_outputs;

            for (int iter_v = k * output_dim; iter_v < (k + 1) * output_dim; iter_v++)
                sample_expected_outputs.push_back(flattened_expected_outputs[iter_v]);

            float* tab_sample_inputs = &sample_inputs[0];
            this->forward_pass(tab_sample_inputs, sample_inputs.size(), is_classification);

            for (int j = 1; j < this->d[L] + 1; j++) {

                this->deltas[L][j] = (this->X[L][j] - sample_expected_outputs[j - 1]);

                if (is_classification == true)
                    this->deltas[L][j] = (1 - this->X[L][j] * this->X[L][j]) * this->deltas[L][j];

            }

            for (int l = L; l >= 1; l--) {

                for (int i = 0; i < this->d[l - 1] + 1; i++) {

                    float sum_result = 0.0;

                    for (int j = 1; j < this->d[l] + 1; j++)
                        sum_result += this->W[l][i][j] * this->deltas[l][j];

                    this->deltas[l - 1][i] = (1 - this->X[l - 1][i] * this->X[l - 1][i]) * sum_result;

                }

            }

            for (int l = 1; l < L + 1; l++) {

                for (int i = 0; i < this->d[l - 1] + 1; i++) {

                    for (int j = 1; j < this->d[l] + 1; j++)
                        this->W[l][i][j] += -alpha * this->X[l - 1][i] * this->deltas[l][j];

                }

            }

        }

    }

};

DLLEXPORT void* create_mlp_model(int* a, int size) {

    std::vector<int> d(a, a+size);

    std::vector<std::vector<std::vector<float>>> W;

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0,1.0);

    srand (static_cast <unsigned> (time(0)));

    for (int l = 0; l < d.size(); l++) {

        //W.resize(W.size() + 1);
        std::vector<std::vector<float>> to_push;

        if (l == 0) {

            W.push_back(to_push);
            continue;

        }


        for (int i = 0; i < d[l - 1] + 1; i++) {

            //W[l].resize(W[l].size() + 1);
            std::vector<float> to_push_v;

            for (int j = 0; j < d[l] + 1; j++) {

                float number = distribution(generator);
                to_push_v.push_back(number);

            }

            to_push.push_back(to_push_v);

        }

        W.push_back(to_push);

    }

    std::vector<std::vector<float>> X;

    for (int l = 0; l < d.size(); l++) {

        //X.resize(X.size() + 1);
        std::vector<float> to_push_v2;

        for (int j = 0; j < d[l] + 1; j++) {

            if (j == 0)
                to_push_v2.push_back(1.0);
            else
                to_push_v2.push_back(0.0);

        }

        X.push_back(to_push_v2);

    }

    std::vector<std::vector<float>> deltas;

    for (int l = 0; l < d.size(); l++) {

        //deltas.resize(deltas.size() + 1);
        std::vector<float> to_push_v3;

        for (int j = 0; j < d[l] + 1; j++)
            to_push_v3.push_back(0.0);

        deltas.push_back(to_push_v3);

    }

    auto mlp_model = new MLP(d, W, X, deltas);

    return mlp_model;

}

DLLEXPORT void train_classification_stochastic_backprop_mlp_model(MLP* model,
                                                              float* flattened_dataset_inputs,
                                                              int input_size,
                                                              float* flattened_expected_outputs,
                                                              int output_size,
                                                              float alpha = 0.01,
                                                              int iterations_count = 1000) {

    model->train_stochastic_gradient_backpropagation(flattened_dataset_inputs,
                                                     input_size,
                                                     flattened_expected_outputs,
                                                     output_size,
                                                     true,
                                                     alpha,
                                                     iterations_count);

}

DLLEXPORT void train_regression_stochastic_backprop_mlp_model(MLP* model,
                                                              float* flattened_dataset_inputs,
                                                              int input_size,
                                                              float* flattened_expected_outputs,
                                                              int output_size,
                                                              float alpha = 0.01,
                                                              int iterations_count = 1000) {

    model->train_stochastic_gradient_backpropagation(flattened_dataset_inputs,
                                                     input_size,
                                                     flattened_expected_outputs,
                                                     output_size,
                                                     false,
                                                     alpha,
                                                     iterations_count);

}

DLLEXPORT float* predict_mlp_model_classification(MLP* model, float* sample_inputs, int size) {

    model->forward_pass(sample_inputs, size, true);

    std::vector<float> values = model->X[model->d.size() - 1];
    float* tab_values = &values[0];

    std::vector<float> contain_val;

    for(int i = 1; i < values.size(); i++)
        contain_val.push_back(tab_values[i]);

    float* tab_contain_val = &contain_val[0];

    return tab_contain_val;

}

DLLEXPORT float* predict_mlp_model_regression(MLP* model, float* sample_inputs, int size) {

    model->forward_pass(sample_inputs, size, false);

    float val = model->X[model->d.size() - 1][1];
    std::vector<float> contain_val;
    contain_val.push_back(val);

    float* tab_contain_val = &contain_val[0];

    return tab_contain_val;

}

DLLEXPORT void save_mlp_model(MLP *model, const char* tab_path,const char* model_name) {

    std::string deb_path(tab_path);

    std::string beg = "model_";

    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), model_name, &tstruct);

    std::string filename = beg + buf;
    std::string path = deb_path + "\\" + filename + ".json";

    std::ofstream file;
    file.open(path);

    json object;
    json value;

    value["d"] = model->d;
    value["W"] = model->W;
    value["X"] = model->X;
    value["deltas"] = model->deltas;

    std::string linear_class = "mlp_model";

    object[linear_class] = value;

    file << std::setw(4) << object << std::endl;

    file.close();

}

DLLEXPORT void* load_mlp_model(const char* filename) {

    std::string path(filename);

    std::ifstream  file(path);

    json object;
    file >> object;

    json model_values = object["mlp_model"];

    std::vector<int> model_d = model_values["d"];
    std::vector<std::vector<std::vector<float>>> model_W = model_values["W"];
    std::vector<std::vector<float>> model_X = model_values["X"];
    std::vector<std::vector<float>> model_deltas = model_values["deltas"];

    auto model = new MLP(model_d, model_W, model_X, model_deltas);

    return model;

}

DLLEXPORT void destroy_mlp_model(MLP* model) {

    delete(model);

}
