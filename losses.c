#include "losses.h"

bool loss_mean_squared_error(matrix *output, matrix *expected_output, double *loss){
    if (output == NULL || expected_output == NULL ||
        output->rows != expected_output->rows || output->cols != expected_output->cols){
        return false;
    }

    *loss = 0;
    for(int i = 0; i < output->rows; i++){
        for(int j = 0; j < output->cols; j++){
            *loss += pow(output->values[i][j] - expected_output->values[i][j], 2);
        }
    }
    *loss = *loss/output->rows;
    return true;
}


bool loss_mean_absolute_error(matrix *output, matrix *expected_output, double *loss){
    if (output == NULL || expected_output == NULL ||
        output->rows != expected_output->rows || output->cols != expected_output->cols){
        return false;
    }

    *loss = 0;
    double v;
    for(int i = 0; i < output->rows; i++){
        for(int j = 0; j < output->cols; j++){
            v = output->values[i][j] - expected_output->values[i][j];
            if (v > 0){
                *loss += v;
            }else{
                *loss -= v;
            }
        }
    }
    *loss = *loss/output->rows;

    return true;
}