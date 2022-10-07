#include "fully_connected_layer.h"

bool layer_init(int inputs, int outputs, char activation, layer *layer){
    if (inputs <= 0 || outputs <= 0 || layer == NULL ||
       (activation != 'r' && activation != 's' && activation != 't')){
            return false;
    }

    layer->n_inputs = inputs;
    layer->n_outputs = outputs;
    layer->activation_function = activation;

    matrix weights;
    matrix biases;
    matrix_init(inputs, outputs, &weights);
    matrix_init(1, outputs, &biases);

    layer->weights = weights;
    layer->biases = biases;

    if (!matrix_random_init(0, 1, -1, 3, &layer->weights)) return false;
    if (!matrix_random_init(0, 1, -1, 3, &layer->biases)) return false;

    return true;
}


bool layer_forward(matrix *data, layer *layer){
    if (data == NULL || layer == NULL){
        return false;
    }

    matrix input;
    matrix_init(data->rows, data->cols, &input);
    matrix_copy(data, &input);
    layer->input = input;

    matrix output;
    matrix_init(data->rows, layer->n_outputs, &output);

    matrix_multiplication(data, &layer->weights, &output);
    matrix_sum_column_by_line(&output, &layer->biases, &output);

    if (layer->activation_function == 'r'){
        activation_relu(&output);
    }else if (layer->activation_function == 's'){
        activation_sigmoid(&output);
    }else if (layer->activation_function == 't'){
        activation_tanh(&output);
    }

    layer->output = output;

    return true;

}

/*
matrix layer_backward_last_layer(matrix *network_errors, layer *last_layer, double learning_rate){
    //calculando as derivadas das saídas da camada
    matrix output_derivatives;
    matrix_init(network_errors->rows, network_errors->cols, &output_derivatives);

    matrix_copy(&last_layer->output, &output_derivatives);

    if (last_layer->activation_function == 'r'){
        derivative_relu(&output_derivatives);
    }else if (last_layer->activation_function == 's'){
        derivative_sigmoid(&output_derivatives);
    }else if (last_layer->activation_function == 't'){
        derivative_tanh(&output_derivatives);
    }

    //calculando os deltas multiplicando cada derivada pelo seu erro
    matrix deltas;
    matrix_init(network_errors->rows, last_layer->n_outputs, &deltas);
    matrix_hadamart_product(network_errors, &output_derivatives, &deltas);

    //transpondo o input da camada
    matrix transposed_inputs;
    matrix_init(network_errors->cols, network_errors->rows, &transposed_inputs);
    matrix_transposition(&last_layer->input, &transposed_inputs);

    //calculando a alteração dos pesos
    matrix weights_reductions;
    matrix_init(last_layer->n_inputs, last_layer->n_outputs, &weights_reductions);
    matrix_multiplication(&transposed_inputs, &deltas, &weights_reductions);

    //calculando a alteração dos biases
    matrix biases_reductions;
    matrix_init(1, last_layer->n_outputs, &biases_reductions);
    matrix_sum_columns(&deltas, &biases_reductions);

    //aplicando o learning_rate
    matrix_multiplication_by_constant(&weights_reductions, &weights_reductions, learning_rate);
    matrix_multiplication_by_constant(&biases_reductions, &biases_reductions, learning_rate);

    //alterando os pesos
    matrix_subtraction(&last_layer->weights, &weights_reductions, &last_layer->weights);

    //alterando os biases
    matrix_subtraction(&last_layer->biases, &biases_reductions, &last_layer->biases);

    //calculando o primeiro passo do próximo gradiente
    matrix transposed_weights;
    matrix_init(last_layer->n_outputs, last_layer->n_inputs, &transposed_weights);
    matrix_transposition(&last_layer->weights, &transposed_weights);

    matrix gradient_result;
    matrix_init(deltas.rows, transposed_weights.cols, &gradient_result);
    matrix_multiplication(&deltas, &transposed_weights, &gradient_result);

    //desalocando as variáveis utilizadas durante a função
    matrix_desallocation(&output_derivatives);
    matrix_desallocation(&deltas);
    matrix_desallocation(&transposed_inputs);
    matrix_desallocation(&weights_reductions);
    matrix_desallocation(&biases_reductions);
    matrix_desallocation(&transposed_weights);

    return gradient_result;
}
*/

matrix layer_backward(matrix *last_computed_gradient, layer *layer, double learning_rate){
    //calculando as derivadas das saídas da camada
    matrix output_derivatives;
    matrix_init(last_computed_gradient->rows, last_computed_gradient->cols, &output_derivatives);

    matrix_copy(&layer->output, &output_derivatives);

    if (layer->activation_function == 'r'){
        derivative_relu(&output_derivatives);
    }else if (layer->activation_function == 's'){
        derivative_sigmoid(&output_derivatives);
    }else if (layer->activation_function == 't'){
        derivative_tanh(&output_derivatives);
    }

    //calculando os deltas multiplicando cada derivada pelo seu erro
    matrix deltas;
    matrix_init(last_computed_gradient->rows, layer->n_outputs, &deltas);
    matrix_hadamart_product(last_computed_gradient, &output_derivatives, &deltas);

    //transpondo o input da camada
    matrix transposed_inputs;
    matrix_init(layer->input.cols, layer->input.rows, &transposed_inputs);
    matrix_transposition(&layer->input, &transposed_inputs);

    //calculando a alteração dos pesos
    matrix weights_reductions;
    matrix_init(layer->n_inputs, layer->n_outputs, &weights_reductions);
    matrix_multiplication(&transposed_inputs, &deltas, &weights_reductions);

    //calculando a alteração dos biases
    matrix biases_reductions;
    matrix_init(1, layer->n_outputs, &biases_reductions);
    matrix_sum_columns(&deltas, &biases_reductions);

    //aplicando o learning_rate
    matrix_multiplication_by_constant(&weights_reductions, &weights_reductions, learning_rate);
    matrix_multiplication_by_constant(&biases_reductions, &biases_reductions, learning_rate);

    //alterando os pesos
    matrix_subtraction(&layer->weights, &weights_reductions, &layer->weights);

    //alterando os biases
    matrix_subtraction(&layer->biases, &biases_reductions, &layer->biases);

    //calculando o primeiro passo do próximo gradiente
    matrix transposed_weights;
    matrix_init(layer->n_outputs, layer->n_inputs, &transposed_weights);
    matrix_transposition(&layer->weights, &transposed_weights);

    matrix gradient_result;
    matrix_init(last_computed_gradient->rows, layer->n_inputs, &gradient_result);
    matrix_multiplication(&deltas, &transposed_weights, &gradient_result);

    //desalocando as variáveis utilizadas durante a função
    matrix_desallocation(&output_derivatives);
    matrix_desallocation(&deltas);
    matrix_desallocation(&transposed_inputs);
    matrix_desallocation(&weights_reductions);
    matrix_desallocation(&biases_reductions);
    matrix_desallocation(&transposed_weights);

    return gradient_result;
}
