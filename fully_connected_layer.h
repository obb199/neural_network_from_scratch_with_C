
#ifndef _FULLY_CONECTED_LAYER_H
#define _FULLY_CONECTED_LAYER_H

#include "activations.h"
#include "derivatives.h"

typedef struct{
    int n_inputs;
    int n_outputs;
    char activation_function; //'r' = relu, 's' = sigmoid, 't' = tanh ...
    matrix weights;
    matrix biases;
    matrix input;
    matrix output;
}layer;

bool layer_init(int, int, char, layer *);
bool layer_forward(matrix *, layer *);
matrix layer_backward(matrix *, layer *, double);

#endif // _FULLY_CONECTED_LAYER_H
