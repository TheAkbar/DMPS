import tensorflow as tf
from set_utils import row_wise_mlp

def _rbf_kernel(diff_set):
    with tf.variable_scope("rbf_kernel"):
        sigma = -1*(
            2*tf.get_variable("diffusion_constant", [1], dtype=tf.float32)**2)
        return sigma * diff_set

def pool_on_kernel(kernel, X, op='mean'):
    if op is 'mean':
        return tf.matmul(kernel, X)
    else:
        # kernel is the Nxdxd tensor and X is the Nxdx256 tensor
        kern_normed = tf.einsum('abij,aik->abjk', tf.matrix_diag(kernel), X)
        return tf.reduce_max(kern_normed, axis=2)

class DeepKernel():
    def __init__(self, layer_data, name = "DeepKernel"):
        self.layer_data = layer_data
        self.name = name
    
    def build_kernel_matrix(self, input, row_norm=False, mean_norm=False):
        with tf.variable_scope(self.name):
            set_size = tf.shape(input)[1]

            # input = row_wise_mlp(input, self.layer_data, name="kernel_mlp")
            
            xx = tf.tile(tf.expand_dims(
                tf.reduce_sum(tf.square(input), axis=2), 1
            ), [1, set_size, 1])
            yy = tf.transpose(xx, perm=[0,2,1])
            xy = tf.matmul(input, tf.transpose(input, perm=[0,2,1]))

            kernel = xx + yy - 2*xy
            if mean_norm:
                kernel -= tf.expand_dims(tf.reduce_mean(kernel, axis=2), -1)
            kernel = _rbf_kernel(kernel)
            return tf.nn.softmax(kernel) if row_norm else tf.exp(kernel)

if __name__ == '__main__':
    a = tf.constant([[[1, 2, 3], [400, 5, 6]], [[7, 8, 9], [3, 1, 7]]], dtype=tf.float32)
    k = DeepKernel([
        {"nodes": 4, "sigma": tf.nn.relu},
    ])
    kernel = k.build_kernel_matrix(a, mean_norm=False)
    final = pool_on_kernel(kernel, a, op='max')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(final))
        print(sess.run(kernel))