import numpy as np
from pprint import pprint

# Given a matrix of neuron weights, an input, and an expected output, implement a back propagation function which run sufficiently number of times causes the weights to converge on a solution that produces the expected output from the given input.

layer_width = 10

layer_weight_sets = [
    np.array([[0] * layer_width]),
    np.array([[0] * layer_width] * layer_width),
    np.array([[0] * layer_width] * layer_width),
    np.array([[0] * layer_width] * layer_width),
    np.array([[0]] * layer_width),
]

for weights in layer_weight_sets:
    pprint(weights)
    print(weights.shape)

def forward_propagate_one_layer(previous_layer_values, weight_matrix):
    next_layer_values = []
    for output_cell_weights in weight_matrix:
        output_value = 0
        for previous_value_index, previous_value in enumerate(previous_layer_values):
            weight = output_cell_weights[previous_value_index]
            output_value += weight * previous_value
        next_layer_values.append(output_value)
    return next_layer_values

def calc_forward_propagated_values(input_values, layer_weight_matrices):
    previous_layer_values = input_values
    layer_neuron_values = []
    for weight_matrix in layer_weight_matrices:
        previous_layer_values = forward_propagate_one_layer(previous_layer_values, weight_matrix)
        layer_neuron_values.append(previous_layer_values)

    return layer_neuron_values

def forward_propagate(input_values, layer_weight_matrices):
    layer_neuron_values = calc_forward_propagated_values(input_values, layer_weight_matrices)
    return layer_neuron_values[-1]

def calc_error(input_values, layer_weight_matrices, expected_output_values):
    prediction = forward_propagate(input_values, layer_weight_matrices)
    error = abs(prediction[0] - expected_output_values[0])
    return error

def calc_one_pass_backpropped_weight_matrices(input_values, layer_weight_matrices, expected_output_values):
    # need a matrix of neuron values
    # calc neurons by forward propping
    layer_neuron_values = calc_forward_propagated_values(input_values, layer_weight_matrices)
    predicted_output = layer_neuron_values[-1]
    # a matrix of weights
    # and the expected output
    # then iterate backwards

    new_layer_weight_matrices = []
    # from the last layer:
    for layer_index in range(len(layer_weight_matrices) - 1, -1, -1):
        layer_weight_matrix = layer_weight_matrices[layer_index]
        new_layer_weight_matrix = []
        # for the first output neuron
        for output_neuron_weights in layer_weight_matrix:
            # for the first weight
            new_output_neuron_weights = []
            for weight_index, weight in enumerate(output_neuron_weights):
    #       get that weight's neuron's value
                neuron_value = layer_neuron_values[layer_index][weight_index]
    #       then calc the derivate of error wrt that weight as:
                deriv = -1 * (expected_output_values[0] - predicted_output[0]) * neuron_value
    #           -1 * (actual - prediction) * neuron value
#           then update that weight as its current value - (some learning rate * the derivative)
                new_weight = weight - (0.1 * deriv)
                new_output_neuron_weights.append(new_weight)
            new_layer_weight_matrix.append(new_output_neuron_weights)
        new_layer_weight_matrices.append(new_layer_weight_matrix)

    return new_layer_weight_matrices
# how can we modularize this more? not sure

def calc_backpropped_weights_for_x_passes(input_values, layer_weight_matrices, expected_output_values, num_passes):
    new_layer_weight_matrices = layer_weight_matrices
    for pass_index in range(num_passes):
        new_layer_weight_matrices = calc_one_pass_backpropped_weight_matrices(input_values, new_layer_weight_matrices, expected_output_values)

    return new_layer_weight_matrices