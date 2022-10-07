#include "derivatives.h"


bool derivative_relu(matrix *result){
    if (result == NULL){
        return false;
    }

    for(int i = 0; i < result->rows; i++){
        for (int j = 0; j < result->cols; j++){
            if (result->values[i][j] < 0){
                result->values[i][j] = 0;
            }else{
                result->values[i][j] = 1;
            }
        }
    }

    return true;
}


bool derivative_sigmoid(matrix *result){
    if (result == NULL){
        return false;
    }

    double sigmoid;
    for(int i = 0; i < result->rows; i++){
        for (int j = 0; j < result->cols; j++){
            sigmoid = 1/(1+(pow(2.718281828459, -1*result->values[i][j])));
            result->values[i][j] = sigmoid*(1 - sigmoid);
        }
    }

    return true;
}


bool derivative_tanh(matrix *result){
    if (result == NULL){
        return false;
    }

    for(int i = 0; i < result->rows; i++){
        for (int j = 0; j < result->cols; j++){
            result->values[i][j] = 1 - pow(tanh(result->values[i][j]), 2);
        }
    }

    return true;
}
