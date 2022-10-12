#ifndef _LOSSES_H
#define _LOSSES_H

#include <stdbool.h>
#include "matrix_operations.h"

bool loss_mean_squared_error(matrix *, matrix *, double *);
bool loss_mean_absolute_error(matrix *, matrix *, double *);
bool loss_categorical_crossentropy(matrix *, matrix *, double *);
bool loss_binary_crossentropy(matrix *, matrix *, double *);

#endif //_LOSSES_H
