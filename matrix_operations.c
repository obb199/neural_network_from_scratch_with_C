#include "matrix_operations.h"


double** matrix_allocation(int lines_m1, int cols_m1){
    if (lines_m1 <= 0 || cols_m1 <= 0){
        return NULL;
    }

    double **m1 = malloc(lines_m1 * sizeof(double *));

    if (m1 == NULL) return NULL;

    for (int i = 0; i < lines_m1; i++){
        m1[i] = malloc(cols_m1 * sizeof(double));
        if (m1[i] == NULL) return NULL;
    }

    return m1;
}


bool matrix_desallocation(int lines_m1, int cols_m1, double** m1){
    if (lines_m1 <= 0 || cols_m1 <= 0 || m1 == NULL) return false;

    for(int i = 0; i < lines_m1; i++){
        free(m1[i]);
    }
    free(m1);

    return true;
}



bool matrix_sum(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2, int lines_m3, int cols_m3, double **m3){
    if (lines_m1 <= 0 || cols_m1 <= 0 ||
        lines_m2 <= 0 || cols_m2 <= 0 ||
        lines_m3 <= 0 || cols_m3 <= 0 ||
        lines_m1 != lines_m2 || cols_m1 != cols_m2 ||
        lines_m1 != lines_m3 || cols_m1 != cols_m3 ||
        m1 == NULL || m2 == NULL || m3 == NULL){
        return false;
    }

    for(int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m3[i][j] = m1[i][j] + m2[i][j];
        }
    }

    return true;
}


bool matrix_subtraction(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2, int lines_m3, int cols_m3, double **m3){
    if (lines_m1 <= 0 || cols_m1 <= 0 ||
        lines_m2 <= 0 || cols_m2 <= 0 ||
        lines_m3 <= 0 || cols_m3 <= 0 ||
        lines_m1 != lines_m2 || cols_m1 != cols_m2 ||
        lines_m1 != lines_m3 || cols_m1 != cols_m3 ||
        m1 == NULL || m2 == NULL || m3 == NULL){
        return false;
    }

    for(int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m3[i][j] = m1[i][j] - m2[i][j];
        }
    }

    return true;
}


bool matrix_sum_columns(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2){
    if(lines_m1 <= 0 || cols_m1 <= 0 ||
       lines_m2 != 1|| cols_m1 != cols_m2 ||
       m1 == NULL || m2 == NULL){
        return false;
    }

    if (!matrix_zeros_init(lines_m2, cols_m2, m2)) return false;

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m2[0][j] += m1[i][j];
        }
    }

    return true;
}


bool matrix_sum_column_by_line(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2, int lines_m3, int cols_m3, double **m3){
    if(lines_m1 <= 0 || cols_m1 <= 0 ||
       lines_m2 <= 0 || cols_m2 <= 0 ||
       lines_m1 != lines_m3 ||
       cols_m1 != cols_m3 ||
       lines_m2 != 1 || cols_m2 != cols_m1){
        return false;
    }

    for (int i = 0; i < lines_m1 ; i++){
        for (int j = 0; j < cols_m1; j++){
            m3[i][j] = m1[i][j] + m2[0][j];
        }
    }

    return true;
}


bool matrix_multiplication(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2, int lines_m3, int cols_m3, double **m3){
    if (lines_m1 <= 0 || cols_m1 <= 0 ||
        lines_m2 <= 0 || cols_m2 <= 0 ||
        lines_m3 <= 0 || cols_m3 <= 0 ||
        m1 == NULL || m2 == NULL || m3 == NULL ||
        cols_m1 != lines_m2 || lines_m1 != lines_m3 || cols_m2 != cols_m3){
        return false;
    }

    if (!(matrix_zeros_init(lines_m3, cols_m3, m3))){
         return false;
    }

    for (register int i = 0; i < lines_m1; i++){
        for (register int j = 0; j < cols_m2; j++){
            for (register int k = 0; k < cols_m1; k++){
                m3[i][j] +=  m1[i][k] * m2[k][j];
            }
        }
    }

    return true;
}

bool matrix_multiplication_by_constant(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2, double constant){
    if (lines_m1 <= 0 || cols_m1 <= 0 ||
        m1 == NULL || m2 == NULL ||
        lines_m1 != lines_m2 || cols_m1 != cols_m2){
        return false;
    }

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m2[i][j] = constant*m1[i][j];
        }
    }

    return true;
}


bool matrix_hadamart_product(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2, int lines_m3, int cols_m3, double **m3){
    if (lines_m1 <= 0 || cols_m1 <= 0 ||lines_m2 <= 0 ||
        lines_m1 != lines_m2 || lines_m1 != lines_m3 ||
        cols_m1 != cols_m2 || cols_m1 != cols_m3 ||
        m1 == NULL || m2 == NULL || m3 == NULL){
        return false;
    }

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m3[i][j] = m1[i][j] * m2[i][j];
        }
    }

    return true;
}


bool matrix_transposition(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2){
    if (lines_m1 <= 0 || cols_m1 <= 0 ||
        lines_m2 <= 0 || cols_m2 <= 0 ||
        lines_m1 != cols_m2 || cols_m1 != lines_m2 ||
        m1 == NULL || m2 == NULL){
        return false;
    }

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m2[j][i] = m1[i][j];
        }
    }

    return true;
}


bool matrix_random_init(double min_value, double max_value, int seed, int precision, int lines_m1, int cols_m1, double **m1){
    if (lines_m1 <= 0 || cols_m1 <= 0 || max_value == min_value || m1 == NULL){
        return false;
    }

    if (min_value > max_value){
        int aux = min_value;
        max_value = min_value;
        min_value = aux;
    }

    if (seed != -1){
        srand(seed);
    }else{
        srand(time(NULL));
    }

    double fractional_part_of_min_value = min_value - (int)min_value;
    double fractional_part_of_max_value = max_value - (int)max_value;

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m1[i][j] = rand()%((int)max_value - (int)min_value) + (int)min_value; //integer part
            m1[i][j] += (rand()%(int)(pow(10, precision)))/pow(10, precision) + (fractional_part_of_min_value - fractional_part_of_max_value); //fractional part
        }
    }

    return true;
}


bool matrix_zeros_init(int lines, int cols, double **m1){
    if (lines <= 0 || cols <= 0 || m1 == NULL){
        return false;
    }

    for (int i = 0; i < lines; i++){
        for (int j = 0; j < cols; j++){
            m1[i][j] = 0;
        }
    }

    return true;
}


bool matrix_ones_init(int lines, int cols, double **m1){
    if (lines <= 0 || cols <= 0 || m1 == NULL){
        return false;
    }

    for (int i = 0; i < lines; i++){
        for (int j = 0; j < cols; j++){
            m1[i][j] = 1.0;
        }
    }

    return true;
}


bool matrix_identity_init(int lines_m1, int cols_m1, double **m1){
    if (lines_m1 <= 0 || cols_m1 <= 0 ||
    lines_m1 != cols_m1 || m1 == NULL){
        return false;
    }

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            if (i == j){
                m1[i][j] = 1;
            }else{
                m1[i][j] = 0;
            }
        }
    }

    return true;
}


bool matrix_randomize_lines(int iterations, int seed, int lines_m1, int cols_m1, double **m1){
    if (lines_m1 <= 0 || cols_m1 <= 0 || m1 == NULL || iterations <= 0){
        return false;
    }

    if (seed != -1){
        srand(seed);
    }else{
        srand(time(NULL));
    }

    int line1 = 0, line2 = 0;
    double aux;
    for(int i = 0; i < iterations; i++){
        line1 = rand()%lines_m1;
        line2 = rand()%lines_m1;

        for (int j = 0; j < cols_m1; j++){
            aux = m1[line1][j];
            m1[line1][j] = m1[line2][j];
            m1[line2][j] = aux;
        }
    }

    return true;
}


bool matrix_normalization(int lines_m1, int cols_m1, double **m1){
    if (lines_m1 <= 0 || cols_m1 <= 0 || m1 == NULL){
        return false;
    }

    double max_values_per_column[cols_m1];
    double min_values_per_column[cols_m1];

    for (int i = 0; i < cols_m1; i++){
        max_values_per_column[i] = m1[0][i];
        min_values_per_column[i] = m1[0][i];
    }

    for(int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            if (max_values_per_column[j] < m1[i][j]) max_values_per_column[j] = m1[i][j];
            if (min_values_per_column[j] > m1[i][j]) min_values_per_column[j] = m1[i][j];
        }
    }

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m1[i][j] = (m1[i][j] - min_values_per_column[j])/(max_values_per_column[j] - min_values_per_column[j]);
        }
    }

    return true;
}


bool matrix_copy(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2){
    if (lines_m1 <= 0 || cols_m1 <= 0 || lines_m2 <= 0 || cols_m2 <= 0 || m1 == NULL || m2 == NULL || lines_m1 != lines_m2 || cols_m1 != cols_m2){
        return false;
    }

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            m2[i][j] = m1[i][j];
        }
    }

    return true;
}


bool matrix_print(int lines_m1, int cols_m1, double **m1){
    if (lines_m1 <= 0 || cols_m1 <= 0 || m1 == NULL){
        return false;
    }

    for (int i = 0; i < lines_m1; i++){
        for (int j = 0; j < cols_m1; j++){
            printf("%.3f ", m1[i][j]);
        }
        printf("\n");
    }

    return true;
}


bool matrix_pointer_verify(double **m1){
    if (m1 == NULL) return false;
    return true;
}


bool matrix_reshape(int lines_m1, int cols_m1, double **m1, int lines_m2, int cols_m2, double **m2){
    if (lines_m1 <= 0 || cols_m1 <= 0 ||
        lines_m2 <= 0 || cols_m2 <= 0 ||
        m1 == NULL || m2 == NULL ||
        lines_m1*cols_m1 != lines_m2*cols_m2){
         return false;
    }

    int actual_line = 0;
    int actual_col = 0;

    for(int i = 0; i < lines_m1; i++){
        for(int j = 0; j < cols_m1; j++){
            m2[actual_line][actual_col] = m1[i][j];
            if (actual_col == cols_m2-1){
                actual_col = 0;
                actual_line += 1;
            }else{
                actual_col++;
            }
        }
    }

    return true;
}
