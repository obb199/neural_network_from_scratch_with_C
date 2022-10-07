#include "activations.h"

bool activation_relu(matrix *output_layer){
    if (output_layer == NULL){
        return false;
    }

    for(int i = 0; i < output_layer->rows; i++){
        for (int j = 0; j < output_layer->cols; j++){
            if (output_layer->values[i][j] < 0){
                output_layer->values[i][j] = 0;
            }
        }
    }

    return true;
}


bool activation_sigmoid(matrix *output_layer){
    if (output_layer == NULL){
        return false;
    }

    for(int i = 0; i < output_layer->rows; i++){
        for (int j = 0; j < output_layer->cols; j++){
            output_layer->values[i][j] = 1/(1+(pow(2.718281828459, -output_layer->values[i][j])));
        }
    }

    return true;
}


bool activation_tanh(matrix *output_layer){
    if (output_layer == NULL){
        return false;
    }

    for(int i = 0; i < output_layer->rows; i++){
        for (int j = 0; j < output_layer->cols; j++){
            output_layer->values[i][j] = tanh(output_layer->values[i][j]);
        }
    }

    return true;
}
