#include "losses.h"

bool loss_mean_squared_error(matrix *output, matrix *expected_output, double *loss){
    if (output == NULL || expected_output == NULL ||
        output->rows != expected_output->rows || output->cols != expected_output->cols){
        return false;
    }

    *loss = 0;
    int i,j ;
    for(i = 0; i < output->rows; i++){
        for(j = 0; j < output->cols; j++){
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
    int i, j;
    for(i = 0; i < output->rows; i++){
        for(j = 0; j < output->cols; j++){
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

bool loss_binary_crossentropy(matrix *output, matrix *expected_output, double *loss){
    if (output == NULL || expected_output == NULL ||
        output->rows != expected_output->rows || output->cols != expected_output->cols){
        return false;
    }

    *loss = 0;
    int i;
    for(i = 0; i < output->rows; i++){
        if (output->values[i][0] != 0) {
            *loss += expected_output->values[i][0] * log10(output->values[i][0]);
        }

        if ((1-output->values[i][0]) != 0){
            *loss += (1-expected_output->values[i][0]) * (log10(1-output->values[i][0]));
        }
    }

    *loss = -*loss/output->rows;

    return true;
}

