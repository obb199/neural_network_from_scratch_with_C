#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdio.h>
#include <stdlib.h.>
#include <stdbool.h>
#include <time.h>
#include <math.h>

double** matrix_allocation(int, int);
bool matrix_desallocation(int, int, double**);
bool matrix_sum(int, int, double**, int, int, double**, int, int, double**);
bool matrix_subtraction(int, int, double**, int, int, double**, int, int, double**);
bool matrix_sum_columns(int, int, double**, int, int, double**);
bool matrix_sum_column_by_line(int, int, double**, int, int, double**, int, int, double**);
bool matrix_multiplication(int, int, double**, int, int, double**, int, int, double**);
bool matrix_multiplication_by_constant(int, int, double**, int, int, double**, double);
bool matrix_hadamart_product(int, int, double **, int, int, double**, int, int, double**);
bool matrix_transposition(int, int, double**, int, int, double**);
bool matrix_random_init(double, double, int, int, int, int, double**);
bool matrix_zeros_init(int, int, double**);
bool matrix_ones_init(int, int, double**);
bool matrix_identity_init(int, int, double**);
bool matrix_randomize_lines(int, int, int, int, double**);
bool matrix_normalization(int, int, double**);
bool matrix_copy(int, int, double**, int, int, double**);
bool matrix_print(int, int, double**);
bool matrix_pointer_verify(double**);
bool matrix_reshape(int, int, double**, int, int, double**);

#endif // __MATRIX_H
