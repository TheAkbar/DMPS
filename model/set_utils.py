import tensorflow as tf

def linear_layer(input, in_nodes, nodes, initializer=None, 
                    identity_supplement=False):
    weights = tf.get_variable(
        "mlp_weight", [in_nodes, nodes], tf.float64,
        initializer=initializer,
    )
    bias = tf.get_variable('bias', [nodes], tf.float64, initializer=initializer)
    if identity_supplement and in_nodes == nodes:
        weights.assign(weights + tf.eye(nodes, dtype=tf.float64))
    return tf.einsum('aij,jk->aik', input, weights) + bias # layer of multilayer perceptron


def row_wise_mlp(input, hidden_sizes=[], name="mlp_layer", sigma=tf.nn.tanh,
                identity_supplement=False, initializer=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        while len(input.get_shape()) < 3:
            input = tf.expand_dims(input, 0)
        
        in_nodes = input.get_shape()[2]
        for i, nodes in enumerate(hidden_sizes):
            with tf.variable_scope("layer_{}".format(i)):
                input = sigma(linear_layer(
                    input, in_nodes, nodes, initializer=initializer, 
                    identity_supplement=identity_supplement,
                ))
                in_nodes = nodes
        return input

