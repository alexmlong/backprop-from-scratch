from backprop import calc_backpropped_weights_for_x_passes, calc_one_pass_backpropped_weight_matrices, calc_error, calc_forward_propagated_values, forward_propagate, forward_propagate_one_layer, train_on_one_sample

def test_forward_propagate_one_layer():
    previous_layer_values = [0, 0, 0, 9]
    weight_matrix = [
        [4, 4, 4, 4],
        [0, 0, 0, 9],
        [4, 4, 4, 4],
        [4, 4, 4, 4],
    ]
    expected_next_layer_values = [36, 81, 36, 36]

    next_layer_values = forward_propagate_one_layer(previous_layer_values, weight_matrix)
    assert expected_next_layer_values == next_layer_values

def test_calc_forward_propagated_values():
    input_values = [2]
    layer_weight_matrices = [
        [
            [1],
            [0],
            [2],
        ],
        [
            [3, 2, 1],
        ]
    ]
    layer_neuron_values = calc_forward_propagated_values(input_values, layer_weight_matrices)
    assert layer_neuron_values == [
        [2, 0, 4],
        [10]
    ]

def test_forward_propagate():
    input_values = [2]
    layer_weight_matrices = [
        [
            [1],
            [0],
            [2],
        ],
        [
            [1, 1, 0],
            [0, 1, 0],
            [2, 1, 1],
        ],
        [
            [3, 2, 1],
        ]
    ]

    # 1st hidden layer values:
    # 2, 0, 4
    # 2nd hidden layer values:
    # 2, 0, 8

    expected_output = [6 + 0 + 8]

    output = forward_propagate(input_values, layer_weight_matrices)
    assert expected_output == output

def test_calc_error():
    input_values = [3]
    layer_weight_matrices = [[[1]]]
    expected_output_values = [9]
    expected_error = 6
    error = calc_error(input_values, layer_weight_matrices, expected_output_values)
    assert expected_error == error

# def test_calc_deriv_wrt_weight():
#     input_values = [2]

#     layer_weight_matrices = [
#         [
#             [1],
#             [0],
#             [2],
#         ],
#         [
#             [1, 1, 0],
#             [0, 1, 0],
#             [2, 1, 1],
#         ],
#         [
#             [3, 2, 1],
#         ]
#     ]

#     predicted_output = [13]
#     expected_output = [14]

#     deriv = calc_deriv_wrt_weight(layer_weight_matrices, predicted_output, expected_output, layer_index = 2, weight_index = 0)
#     assert deriv == 


def test_calc_one_pass_backpropped_weight_matrices():
    input_values = [2]

    initial_layer_weight_matrices = [
        [
            [0.1],
        ]
    ]

    initial_predicted_output = [8]
    expected_output = [10]

    updated_weight_matrices = calc_one_pass_backpropped_weight_matrices(input_values, initial_layer_weight_matrices, expected_output)

    assert updated_weight_matrices[0][0] > initial_layer_weight_matrices[0][0]

def test_calc_backpropped_weights_for_x_passes():
    input_values = [2]

    initial_layer_weight_matrices = [
        [
            [0.1],
        ]
    ]

    initial_predicted_output = [8]
    expected_output = [10]

    updated_weight_matrices = calc_backpropped_weights_for_x_passes(input_values, initial_layer_weight_matrices, expected_output, num_passes=1000)

    assert updated_weight_matrices[0][0][0] > 4.9 and updated_weight_matrices[0][0][0] < 5.1