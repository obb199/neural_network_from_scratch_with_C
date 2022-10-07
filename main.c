#include "fully_connected_layer.h"

int main(){
    layer l1;
    layer l2;
    layer l3;

    layer_init(3, 2, 'r', &l1);
    layer_init(2, 2, 'r', &l2);
    layer_init(2, 2, 'r', &l3);

    int batch_size = 16;
    matrix data;
    matrix_init(batch_size, 3, &data);

    for(int i = 0; i < batch_size; i++){
        data.values[i][0] = i;
        data.values[i][1] = i+1;
        data.values[i][2] = i+2;
    }

    matrix_normalization(&data);

    matrix expected_output;
    matrix_init(batch_size, 2, &expected_output);
    for (int i = 0; i < batch_size; i++){
        expected_output.values[i][0] = data.values[i][0] + data.values[i][1] + data.values[i][2] + 1;
        expected_output.values[i][1] = data.values[i][0]*2 + data.values[i][1]*3 + data.values[i][2]*2 + 2;
    }

    matrix grad1;
    matrix grad2;
    matrix errors;
    matrix_init(batch_size, 2, &errors);

    float learning_rate = 0.002;
    for (int i = 0; i < 500; i++){
        //feedforward
        layer_forward(&data, &l1);
        layer_forward(&l1.output, &l2);
        layer_forward(&l2.output, &l3);

        //error calculation
        matrix_subtraction(&l3.output, &expected_output, &errors);

        //backpropagation
        grad1 = layer_backward(&errors, &l3, learning_rate);
        grad2 = layer_backward(&grad1, &l2, learning_rate);
        layer_backward(&grad2, &l1, learning_rate);

    }

    matrix_print(&expected_output);
    printf("\n");
    matrix_print(&l3.output);

    return 0;
}
