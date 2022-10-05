#include "fully_connected_layer.h"
#include "matrix_operations.h"

int main(){
    int batch_size = 128;
    int n_inputs = 2;
    int n_outputs = 3;

    //criando as camadas
    struct layer first_layer;
    struct layer second_layer;
    struct layer third_layer;

    //criando os dados - pontos de uma reta
    double **data = NULL;
    double **expected_outputs = NULL;

    data = matrix_allocation(batch_size, n_inputs);
    expected_outputs = matrix_allocation(batch_size, n_outputs);

    //valores de entrada
    for (int i = 0; i < batch_size; i++){
        data[i][0] = i;
        data[i][1] = i*2;
    }

    matrix_normalization(batch_size, n_inputs, data);

    //valores de saida
    for (int i = 0; i < batch_size; i++){
        expected_outputs[i][0] = data[i][0]*2 + data[i][1]*3 + 4;
        expected_outputs[i][1] = data[i][0] + data[i][1] + 1;
        expected_outputs[i][2] = data[i][0]*5 + data[i][1]*2 + 2;
    }

    //criando as  camadas
    if (!layer_init(n_inputs, 4, 'r', &first_layer)) return -1;
    if (!layer_init(4, 4, 'r', &second_layer)) return -2;
    if (!layer_init(4, n_outputs, 'r', &third_layer)) return -3;

    //forward
    if (!layer_forward(batch_size, data, &first_layer)) return-4;
    if (!layer_forward(batch_size, first_layer.output, &second_layer)) return -5;
    if (!layer_forward(batch_size, second_layer.output, &third_layer)) return -6;

    //errors
    double  **errors = NULL;
    errors = matrix_allocation(batch_size, n_outputs);

    //loop de treino com feedforward e backpropagation
    //colocar dentro de layer
    double** first_gradient = NULL;
    double **second_gradient = NULL;
    double **third_gradient = NULL;

    matrix_subtraction(batch_size, third_layer.n_outputs, third_layer.output,
                                     batch_size, third_layer.n_outputs, expected_outputs,
                                     batch_size, third_layer.n_outputs, errors);

    matrix_print(batch_size, third_layer.n_outputs, errors);

    for (int i = 0; i < 15000; i++){
        first_gradient = layer_backward_last_layer(batch_size, errors, 0.0001, &third_layer);
        second_gradient = layer_backward_general(batch_size, 0.0001, &second_layer, first_gradient);
        third_gradient = layer_backward_general(batch_size, 0.0001, &third_layer, second_gradient);

        matrix_subtraction(batch_size, third_layer.n_outputs, third_layer.output,
                                     batch_size, third_layer.n_outputs, expected_outputs,
                                     batch_size, third_layer.n_outputs, errors);

        //matrix_print(batch_size, third_layer.n_outputs, errors);

        layer_forward(batch_size, data, &first_layer);
        layer_forward(batch_size, first_layer.output, &second_layer);
        layer_forward(batch_size, second_layer.output, &third_layer);
    }

    printf("\n\n\n\n");
    matrix_print(batch_size, third_layer.n_outputs, third_layer.output);

    return 0;
}

