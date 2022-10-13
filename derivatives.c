#include "derivatives.h"


bool derivative_relu(matrix *result){
    if (result == NULL){
        return false;
    }

    int i, j;
    for(i = 0; i < result->rows; i++){
        for (j = 0; j < result->cols; j++){
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
    int i, j;
    for(i = 0; i < result->rows; i++){
        for (j = 0; j < result->cols; j++){
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

    int i, j;
    for(i = 0; i < result->rows; i++){
        for (j = 0; j < result->cols; j++){
            result->values[i][j] = 1 - pow(tanh(result->values[i][j]), 2);
        }
    }

    return true;
}


bool derivative_loss_mean_squared_error(matrix *output, matrix *expected_output, matrix *gradient_matrix){
    if (expected_output == NULL || output == NULL || gradient_matrix == NULL ||
        expected_output->rows != output->rows || expected_output->rows != gradient_matrix->rows ||
        expected_output->cols != output->cols || expected_output->cols != gradient_matrix->cols){
        return false;
    }

    matrix_subtraction(output, expected_output, gradient_matrix);
    matrix_multiplication_by_constant(gradient_matrix, gradient_matrix, 2.0/gradient_matrix->rows);

    return true;
}

bool derivative_loss_absolute_error(matrix *output, matrix *expected_output, matrix *gradient_matrix){
    if (expected_output == NULL || output == NULL || gradient_matrix == NULL ||
        expected_output->rows != output->rows || expected_output->rows != gradient_matrix->rows ||
        expected_output->cols != output->cols || expected_output->cols != gradient_matrix->cols){
        return false;
    }

    for(int i = 0; i < gradient_matrix->rows; i++){
        for (int j = 0; j < gradient_matrix->cols; j++){
            if (output->values[i][j] >= expected_output->values[i][j]){
                gradient_matrix->values[i][j] = 1;
            }else{
                gradient_matrix->values[i][j] = -1;
            }
        }
    }

    return true;
}

bool derivative_loss_binary_crossentropy(matrix *output, matrix *expected_output, matrix *gradient_matrix) {
    if (expected_output == NULL || output == NULL || gradient_matrix == NULL ||
        expected_output->rows != output->rows || expected_output->rows != gradient_matrix->rows ||
        expected_output->cols != output->cols || expected_output->cols != gradient_matrix->cols) {
        return false;
    }

    int i;
    for (i = 0; i < output->rows; i++){
        gradient_matrix->values[i][0] = expected_output->values[i][0] - output->values[i][0];
    }

    return true;
}