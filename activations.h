#ifndef _ACTIVATIONS_H
#define _ACTIVATIONS_H

#include "matrix_operations.h"

bool activation_relu(matrix *);
bool activation_sigmoid(matrix *);
bool activation_tanh(matrix *);
bool activation_softmax(matrix *);

#endif // _ACTIVATIONS_H
