import tensorflow as tf

def row_wise_mlp(input, layer_data, name = "mlp_layer", 
            identity_supplement=False, mat=False, init_pos=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if mat:
            in_nodes = input.get_shape()[1]
        else:
            num_sets, set_size = (tf.shape(input)[0], tf.shape(input)[1])
            in_nodes = input.get_shape()[2]
            input = tf.reshape(input, [num_sets*set_size, in_nodes])
        for i, data in enumerate(layer_data):
            with tf.variable_scope("layer_{}".format(i)):
                nodes, sigma = (data['nodes'], data['sigma'])
                if sigma == tf.nn.tanh:
                    initializer = tf.contrib.layers.xavier_initializer()
                else:
                    initializer = None
                weights = tf.get_variable(
                    "mlp_weight", [in_nodes, nodes], tf.float32,
                    initializer=initializer,
                )
                bias = tf.get_variable('bias', [nodes], tf.float32, initializer=initializer)

                if init_pos:
                    weights.assign(tf.abs(weights)) 
                    bias.assign(tf.abs(bias))
                
                if identity_supplement and in_nodes == nodes:
                    weights.assign(weights + tf.eye(nodes))
                input = sigma(tf.matmul(input, weights)) + bias # layer of multilayer perceptron
                in_nodes = nodes
        return tf.reshape(input, [num_sets, set_size, in_nodes]) if not mat else input

