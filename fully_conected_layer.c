#include "fully_connected_layer.h"

bool layer_init(int inputs, int outputs, char activation, struct layer * actual_layer){
    if (inputs <= 0 || outputs <= 0 || actual_layer == NULL ||
       (activation != 'r' && activation != 's' && activation != 't')){
            return false;
    }

    actual_layer->n_inputs = inputs;
    actual_layer->n_outputs = outputs;
    actual_layer->activation_function = activation;

    actual_layer->weights = matrix_allocation(inputs, outputs);
    actual_layer->biases = matrix_allocation(1, outputs);

    actual_layer->input = NULL;
    actual_layer->output = NULL;

    if (!matrix_random_init(0, 1, -1, 5, actual_layer->n_inputs, actual_layer->n_outputs, actual_layer->weights)) return false;
    if (!matrix_random_init(0, 1, -1, 5, 1, actual_layer->n_outputs, actual_layer->biases)) return false;

    return true;
}


bool layer_forward(int batch_size, double **data, struct layer * layer){
    if (batch_size <= 0 || data == NULL || layer == NULL){
        return false;
    }

    if (layer->input == NULL && layer->output == NULL){
        layer->input = matrix_allocation(batch_size, layer->n_inputs);
        layer->output = matrix_allocation(batch_size, layer->n_outputs);
    }

    matrix_copy(batch_size, layer->n_inputs, data,
                       batch_size, layer->n_inputs, layer->input);

    matrix_multiplication(batch_size, layer->n_inputs, data,
                                  layer->n_inputs, layer->n_outputs, layer->weights,
                                  batch_size, layer->n_outputs, layer->output);

    matrix_sum_column_by_line(batch_size, layer->n_outputs, layer->output,
                                              1, layer->n_outputs, layer->biases,
                                              batch_size, layer->n_outputs, layer->output);


    if (layer->activation_function == 'r'){
        activation_relu(batch_size, layer->n_outputs, layer->output);
    }else if (layer->activation_function == 's'){
        activation_sigmoid(batch_size, layer->n_outputs, layer->output);
    }else if (layer->activation_function == 't'){
        activation_tanh(batch_size, layer->n_outputs, layer->output);
    }

    return true;

}


double** layer_backward_last_layer(int batch_size, double** network_errors, double learning_rate, struct layer * last_layer){
    if (batch_size <= 0 || network_errors == NULL || learning_rate <= 0 || last_layer == NULL) return NULL;

    //calculando as derivadas das saídas da camada
    double** output_derivatives = NULL;
    output_derivatives = matrix_allocation(batch_size, last_layer->n_outputs);

    matrix_copy(batch_size, last_layer->n_outputs, last_layer->output,
                       batch_size, last_layer->n_outputs, output_derivatives);

    //**********************colocar as outras derivadas***********************//
    if (last_layer->activation_function == 'r'){
        derivative_relu(batch_size, last_layer->n_outputs, output_derivatives);
    }else if (last_layer->activation_function == 's'){
        derivative_sigmoid(batch_size, last_layer->n_outputs, output_derivatives);
    }else if (last_layer->activation_function == 't'){
        derivative_tanh(batch_size, last_layer->n_outputs, output_derivatives);
    }

    //calculando os deltas multiplicando cada derivada pelo seu erro
    double** deltas = NULL;
    deltas = matrix_allocation(batch_size, last_layer->n_outputs);
    matrix_hadamart_product(batch_size, last_layer->n_outputs, network_errors,
                                          batch_size, last_layer->n_outputs, output_derivatives,
                                          batch_size, last_layer->n_outputs, deltas);


    //transpondo o input da camada
    double** transposed_inputs = NULL;
    transposed_inputs = matrix_allocation(last_layer->n_inputs, batch_size);
    matrix_transposition(batch_size, last_layer->n_inputs, last_layer->input,
                                  last_layer->n_inputs, batch_size, transposed_inputs);

    //calculando a alteração dos pesos
    double** weights_reductions = NULL;
    weights_reductions = matrix_allocation(last_layer->n_inputs, last_layer->n_outputs);

    matrix_multiplication(last_layer->n_inputs, batch_size, transposed_inputs,
                                  batch_size, last_layer->n_outputs, deltas,
                                  last_layer->n_inputs, last_layer->n_outputs, weights_reductions);


    //calculando a alteração dos biases
    double** biases_reductions = NULL;
    biases_reductions = matrix_allocation(1, last_layer->n_outputs);

    matrix_sum_columns(batch_size, last_layer->n_outputs, deltas,
                                   1, last_layer->n_outputs, biases_reductions);

    //aplicando o learning_rate
    matrix_multiplication_by_constant(last_layer->n_inputs, last_layer->n_outputs, weights_reductions,
                                                      last_layer->n_inputs, last_layer->n_outputs, weights_reductions,
                                                      learning_rate);
    matrix_multiplication_by_constant(1, last_layer->n_outputs, biases_reductions,
                                                      1, last_layer->n_outputs, biases_reductions, learning_rate);

    //alterando os pesos
    matrix_subtraction(last_layer->n_inputs, last_layer->n_outputs, last_layer->weights,
                               last_layer->n_inputs, last_layer->n_outputs, weights_reductions,
                               last_layer->n_inputs, last_layer->n_outputs, last_layer->weights);

    //alterando os biases
    matrix_subtraction(1, last_layer->n_outputs, last_layer->biases,
                               1, last_layer->n_outputs, biases_reductions,
                               1, last_layer->n_outputs, last_layer->biases);

    double **transposed_weights = NULL;
    transposed_weights = matrix_allocation(last_layer->n_outputs, last_layer->n_inputs);
    matrix_transposition(last_layer->n_inputs, last_layer->n_outputs, last_layer->weights,
                                  last_layer->n_outputs, last_layer->n_inputs, transposed_weights);

    double **gradient_result = NULL;
    gradient_result = matrix_allocation(batch_size, last_layer->n_inputs);

    matrix_multiplication(batch_size, last_layer->n_outputs, deltas,
                                  last_layer->n_outputs, last_layer->n_inputs, transposed_weights,
                                  batch_size, last_layer->n_inputs, gradient_result);

    //desalocando as variáveis utilizadas durante a função
    matrix_desallocation(batch_size, last_layer->n_outputs, output_derivatives);
    matrix_desallocation(batch_size, last_layer->n_outputs, deltas);
    matrix_desallocation(last_layer->n_inputs, batch_size, transposed_inputs);
    matrix_desallocation(last_layer->n_inputs, last_layer->n_outputs, weights_reductions);
    matrix_desallocation(1, last_layer->n_outputs, biases_reductions);
    matrix_desallocation(last_layer->n_outputs, last_layer->n_inputs, transposed_weights);

    return gradient_result;
}

double** layer_backward_general(int batch_size, double learning_rate, struct layer * actual_layer, double** last_computed_gradient){
    if (batch_size <= 0 || learning_rate <= 0 || actual_layer == NULL || last_computed_gradient == NULL) return false;

    //calculando as derivadas das saídas da camada
    double** output_derivatives = NULL;
    output_derivatives = matrix_allocation(batch_size, actual_layer->n_outputs);

    matrix_copy(batch_size, actual_layer->n_outputs, actual_layer->output,
                     batch_size, actual_layer->n_outputs, output_derivatives);

    //**********************colocar as outras derivadas***********************//
    if (actual_layer->activation_function == 'r'){
        derivative_relu(batch_size, actual_layer->n_outputs, output_derivatives);
    }else if (actual_layer->activation_function == 's'){
        derivative_sigmoid(batch_size, actual_layer->n_outputs, output_derivatives);
    }else if (actual_layer->activation_function == 't'){
        derivative_tanh(batch_size, actual_layer->n_outputs, output_derivatives);
    }

    //calculando os deltas multiplicando cada derivada pelo seu erro
    double** deltas = NULL;
    deltas = matrix_allocation(batch_size, actual_layer->n_outputs);
    matrix_hadamart_product(batch_size, actual_layer->n_outputs, last_computed_gradient,
                                          batch_size, actual_layer->n_outputs, output_derivatives,
                                          batch_size, actual_layer->n_outputs, deltas);

    //transpondo o input da camada
    double** transposed_inputs = NULL;
    transposed_inputs = matrix_allocation(actual_layer->n_inputs, batch_size);
    matrix_transposition(batch_size, actual_layer->n_inputs, actual_layer->input,
                                  actual_layer->n_inputs, batch_size, transposed_inputs);

    //calculando a alteração dos pesos
    double** weights_reductions = NULL;
    weights_reductions = matrix_allocation(actual_layer->n_inputs, actual_layer->n_outputs);

    matrix_multiplication(actual_layer->n_inputs, batch_size, transposed_inputs,
                                  batch_size, actual_layer->n_outputs, deltas,
                                  actual_layer->n_inputs, actual_layer->n_outputs, weights_reductions);

    //calculando a alteração dos biases
    double** biases_reductions = NULL;
    biases_reductions = matrix_allocation(1, actual_layer->n_outputs);

    matrix_sum_columns(batch_size, actual_layer->n_outputs, deltas,
                                   1, actual_layer->n_outputs, biases_reductions);

    //aplicando o learning_rate
    matrix_multiplication_by_constant(actual_layer->n_inputs, actual_layer->n_outputs, weights_reductions,
                                                      actual_layer->n_inputs, actual_layer->n_outputs, weights_reductions,
                                                      learning_rate);
    matrix_multiplication_by_constant(1, actual_layer->n_outputs, biases_reductions,
                                                      1, actual_layer->n_outputs, biases_reductions, learning_rate);

    //alterando os pesos
    matrix_subtraction(actual_layer->n_inputs, actual_layer->n_outputs, actual_layer->weights,
                               actual_layer->n_inputs, actual_layer->n_outputs, weights_reductions,
                               actual_layer->n_inputs, actual_layer->n_outputs, actual_layer->weights);

    //alterando os biases
    matrix_subtraction(1, actual_layer->n_outputs, actual_layer->biases,
                               1, actual_layer->n_outputs, biases_reductions,
                               1, actual_layer->n_outputs, actual_layer->biases);

    double **transposed_weights = NULL;
    transposed_weights = matrix_allocation(actual_layer->n_outputs, actual_layer->n_inputs);
    matrix_transposition(actual_layer->n_inputs, actual_layer->n_outputs, actual_layer->weights,
                                  actual_layer->n_outputs, actual_layer->n_inputs, transposed_weights);

    double **new_gradient = NULL;
    new_gradient = matrix_allocation(batch_size, actual_layer->n_inputs);

    matrix_multiplication(batch_size, actual_layer->n_outputs, deltas,
                                  actual_layer->n_outputs, actual_layer->n_inputs, transposed_weights,
                                  batch_size, actual_layer->n_inputs, new_gradient);

    //desalocando as variáveis utilizadas durante a função
    matrix_desallocation(batch_size, actual_layer->n_outputs, output_derivatives);
    matrix_desallocation(batch_size, actual_layer->n_outputs, deltas);
    matrix_desallocation(actual_layer->n_inputs, batch_size, transposed_inputs);
    matrix_desallocation(actual_layer->n_inputs, actual_layer->n_outputs, weights_reductions);
    matrix_desallocation(1, actual_layer->n_outputs, biases_reductions);
    matrix_desallocation(actual_layer->n_outputs, actual_layer->n_inputs, transposed_weights);

    return new_gradient;
}

bool layer_description(struct layer * l){
    if (l == NULL) return false;

    printf("n_inputs: %d\n", l->n_inputs);
    printf("n_outputs: %d\n", l->n_outputs);
    printf("activation function: %c\n", l->activation_function);
    printf("weights: \n");
    matrix_print(l->n_outputs, l->n_inputs, l->weights);
    printf("biases: \n");
    matrix_print(1, l->n_outputs, l->biases);
    printf("\n");

    return true;
}
