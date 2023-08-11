from backprop import forward_propagate, forward_propagate_one_layer

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