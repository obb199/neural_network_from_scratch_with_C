#include "activations.h"

bool activation_relu(int lines, int cols, double **matrix){
    if (lines <= 0 || cols <= 0 || matrix == NULL){
        return false;
    }

    for(int i = 0; i < lines; i++){
        for (int j = 0; j < cols; j++){
            if (matrix[i][j] < 0){
                matrix[i][j] = 0;
            }
        }
    }

    return true;
}


bool derivative_relu(int lines, int cols, double **matrix){
    if (lines <= 0 || cols <= 0 || matrix == NULL){
        return false;
    }

    for(int i = 0; i < lines; i++){
        for (int j = 0; j < cols; j++){
            if (matrix[i][j] < 0){
                matrix[i][j] = 0;
            }else{
                matrix[i][j] = 1;
            }
        }
    }

    return true;
}


bool activation_sigmoid(int lines, int cols, double **matrix){
    if (lines <= 0 || cols <= 0 || matrix == NULL){
        return false;
    }

    double sigmoid;
    for(int i = 0; i < lines; i++){
        for (int j = 0; j < cols; j++){
            matrix[i][j] = 1/(1+(pow(2.718281828459, -matrix[i][j])));
        }
    }

    return true;
}


bool derivative_sigmoid(int lines, int cols, double **matrix){
    if (lines <= 0 || cols <= 0 || matrix == NULL){
        return false;
    }

    double sigmoid;
    for(int i = 0; i < lines; i++){
        for (int j = 0; j < cols; j++){
            sigmoid = 1/(1+(pow(2.718281828459, -matrix[i][j])));
            matrix[i][j] = sigmoid*(1 - sigmoid);
        }
    }

    return true;
}


bool activation_tanh(int lines, int cols, double **matrix){
    if (lines <= 0 || cols <= 0 || matrix == NULL){
        return false;
    }

    for(int i = 0; i < lines; i++){
        for (int j = 0; j < cols; j++){
            matrix[i][j] = tanh(matrix[i][j]);
        }
    }

    return true;
}


bool derivative_tanh(int lines, int cols, double **matrix){
    if (lines <= 0 || cols <= 0 || matrix == NULL){
        return false;
    }

    for(int i = 0; i < lines; i++){
        for (int j = 0; j < cols; j++){
            matrix[i][j] = 1 - pow(tanh(matrix[i][j]), 2);
        }
    }

    return true;
}
