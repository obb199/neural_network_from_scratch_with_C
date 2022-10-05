#ifndef _ACTIVATIONS_H
#define _ACTIVATIONS_H

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

bool activation_relu(int, int, double**);
bool derivative_relu(int, int, double**);
bool activation_sigmoid(int, int, double**);
bool derivative_sigmoid(int, int, double**);
bool activation_tanh(int, int, double**);
bool derivative_tanh(int, int, double**);

#endif // _ACTIVATIONS_H
