#ifndef _DERIVATIVES_H
#define  _DERIVATIVES_H

#include "matrix_operations.h"

bool derivative_relu(matrix *);
bool derivative_sigmoid(matrix *);
bool derivative_tanh(matrix *);

#endif //  _DERIVATIVES_H
