#ifndef _DERIVATIVES_H
#define  _DERIVATIVES_H

#include "matrix_operations.h"

bool derivative_relu(matrix *);
bool derivative_sigmoid(matrix *);
bool derivative_tanh(matrix *);
//bool derivative_softmax(matrix *);

bool derivative_loss_mean_squared_error(matrix *, matrix *, matrix *);
bool derivative_loss_absolute_error(matrix *, matrix *, matrix*);
bool derivative_loss_binary_crossentropy(matrix *, matrix *, matrix *);
//bool derivative_loss_categorical_crossentropy(matrix *, matrix *);

#endif //  _DERIVATIVES_H
