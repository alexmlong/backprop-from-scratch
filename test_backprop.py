from backprop import forward_propagate_one_layer


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

