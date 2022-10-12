#include "fully_connected_layer.h"
#include "losses.h"

int main(){
    layer l1;
    layer l2;
    layer l3;

    layer_init(3, 8, 'r', &l1);
    layer_init(8, 8, 'r', &l2);
    layer_init(8, 2, 'r', &l3);

    int batch_size = 128;
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
    matrix derivative_loss;
    matrix_init(batch_size, 2, &derivative_loss);

    double loss_mse, loss_mae;
    double learning_rate = 0.00001;
    for (int i = 0; i < 1000; i++){
        //feedforward
        layer_forward(&data, &l1);
        layer_forward(&l1.output, &l2);
        layer_forward(&l2.output, &l3);

        //error calculation
        derivative_loss_absolute_error(&l3.output, &expected_output, &derivative_loss);

        //backpropagation
        grad1 = layer_backward(&derivative_loss, &l3, learning_rate);
        grad2 = layer_backward(&grad1, &l2, learning_rate);
        layer_backward(&grad2, &l1, learning_rate);

        loss_mean_squared_error(&l3.output, &expected_output, &loss_mse);
        loss_mean_absolute_error(&l3.output, &expected_output, &loss_mae);

        printf("epoch: %d\nmean squared error loss: %lf\nabsolute error loss: %lf\n\n", i+1, loss_mse, loss_mae);
    }

    return 0;
}
