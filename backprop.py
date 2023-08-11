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
    output_layer_values = []
    for output_cell_weights in weight_matrix:
        output_value = 0
        for previous_value_index, previous_value in enumerate(previous_layer_values):
            weight = output_cell_weights[previous_value_index]
            output_value += weight * previous_value
        output_layer_values.append(output_value)
    return output_layer_values
