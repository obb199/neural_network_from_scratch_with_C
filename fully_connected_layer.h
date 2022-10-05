#ifndef _FULLY_CONECTED_LAYER_H
#define _FULLY_CONECTED_LAYER_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include "matrix_operations.h"
#include "activations.h"

struct layer{
    int n_inputs;
    int n_outputs;
    char activation_function; //'r' = relu, 's' = sigmoid, 't' = tanh ...
    double **weights;
    double **biases;
    double **input;
    double **output;
};

bool layer_init(int, int, char, struct layer *);
bool layer_forward(int, double**, struct layer *);

double** layer_backward_last_layer(int batch_size, double** errors, double learning_rate, struct layer * actual_layer);
double** layer_backward_general(int batch_size, double learning_rate, struct layer * actual_layer, double** last_computed_gradient);
bool layer_description(struct layer *);

#endif // _FULLY_CONECTED_LAYER_H
