import tensorflow as tf
from set_utils import row_wise_mlp
import numpy as np

def _rbf_kernel(diff_set):
    with tf.variable_scope("rbf_kernel"):
        sigma = -1*(
            2*tf.get_variable("diffusion_constant", [1], dtype=tf.float64)**2)
        return sigma * diff_set

def pool_on_kernel(kernel, X, op='mean'):
    if op is 'mean':
        return tf.matmul(kernel, X)
    elif op is 'approx_max':
        eps = tf.constant(value=1e-300, dtype=tf.float64)
        scale = 20.0
        val = tf.matmul(tf.pow(kernel, scale)+eps, tf.pow(X, scale)+eps)
        sign = tf.divide(val, tf.abs(val))
        return sign * tf.pow(sign * val, 1/scale)
    else:
        # kernel is the Nxdxd tensor and X is the Nxdx256 tensor
        # kern_normed = tf.einsum('abij,aik->abjk', tf.matrix_diag(kernel), X)
        kern_normed = tf.multiply(tf.expand_dims(kernel, -1), tf.expand_dims(X, 2))
        return tf.reduce_max(kern_normed, axis=1)

class DeepKernel():
    def __init__(self, hidden_sizes, sigma=tf.nn.tanh, name="DeepKernel"):
        self.hidden_sizes = hidden_sizes
        self.name = name
        self.sigma = sigma
    
    def build_kernel_matrix(self, input, row_norm=True, mean_norm=True):
        with tf.variable_scope(self.name):
            set_size = tf.shape(input)[1]
            input = row_wise_mlp(
                input, hidden_sizes=self.hidden_sizes, name="kernel_mlp",
                sigma=self.sigma, initializer=tf.contrib.layers.xavier_initializer(),
            )
            
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
    a = tf.constant([[[1, -2, 3], [-4, 5, 6]], [[7, -8, 9], [3, 1, 7]]], dtype=tf.float64)
    a = a * 1e-8
    k = DeepKernel(hidden_sizes=[4])
    kernel = k.build_kernel_matrix(a, mean_norm=False)
    final = pool_on_kernel(kernel, a, op='max')
    final2 = pool_on_kernel(kernel, a, op='approx_max')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(final))
        print(sess.run(final2))
        print(sess.run(tf.reduce_mean(tf.square(final2-final))))
        print(sess.run(kernel))
        print(sess.run(a))