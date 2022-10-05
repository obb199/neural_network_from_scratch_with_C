#ifndef _DERIVATIVES_H
#define  _DERIVATIVES_H

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

bool derivative_relu(int, int, double**);
bool derivative_sigmoid(int, int, double**);
bool derivative_tanh(int, int, double**);

#endif //  _DERIVATIVES_H

