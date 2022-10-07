#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdio.h>
#include <stdlib.h.>
#include <stdbool.h>
#include <time.h>
#include <math.h>

typedef struct{
    int rows;
    int cols;
    double **values;
}matrix;

double** pointer_allocation(int, int);
bool pointer_desallocation(int, int, double**);
bool matrix_init(int, int, matrix *);
bool matrix_desallocation(matrix *);
bool matrix_print(matrix *);
bool matrix_random_init(double, double, int, int, matrix *);
bool matrix_sum(matrix *, matrix *, matrix *);
bool matrix_subtraction(matrix *, matrix *, matrix *);
bool matrix_sum_columns(matrix *, matrix *);
bool matrix_sum_column_by_line(matrix *, matrix *, matrix *);
bool matrix_multiplication(matrix *, matrix *, matrix *);
bool matrix_multiplication_by_constant(matrix *, matrix *, double);
bool matrix_hadamart_product(matrix *, matrix *, matrix *);
bool matrix_transposition(matrix *, matrix *);
bool matrix_zeros_init(matrix *);
bool matrix_ones_init(matrix *);
bool matrix_identity_init(matrix *);
bool matrix_randomize_lines(int, int, matrix *);
bool matrix_normalization(matrix *);
bool matrix_copy(matrix *, matrix *);
bool matrix_reshape(matrix *, matrix *);

#endif // __MATRIX_H

