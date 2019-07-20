import tensorflow as tf

from deep_kernel import DeepKernel, pool_on_kernel
from set_utils import row_wise_mlp

def block(inputs):
    feature_vec = row_wise_mlp(
        inputs,
        [{"nodes": 256, "sigma": tf.nn.tanh}], 
        name="feature_extraction",
    )
    # learn latent graph
    k = DeepKernel([
        {"nodes": 256, "sigma": tf.nn.tanh},
        {"nodes": 512, "sigma": tf.nn.tanh},
    ])
    learned_graph = k.build_kernel_matrix(feature_vec)
    feature_vec = pool_on_kernel(feature_vec, learned_graph)
    feature_vec = pool_on_kernel(feature_vec, learned_graph)
    feature_vec = pool_on_kernel(feature_vec, learned_graph)
    return feature_vec

class DMPSBlock():
    DMPS = ""
    DENOISE = "denoising_block"
    RESIDUAL = "residual_block"

class MessagePassing():
    def __init__(self, input, dmps_blocks=3):
        self.input = input
        self.dmps_blocks = dmps_blocks

    def get_tf_train_graph(self):
        with tf.variable_scope('DMPS'):
            # feature extraction
            feature_vec = row_wise_mlp(
                self.input,
                [{"nodes": 256, "sigma": tf.nn.relu}], 
                name="feature_extraction", init_pos=True,
            )
            # learn latent graph
            k = DeepKernel([
                {"nodes": 256, "sigma": tf.nn.tanh},
                {"nodes": 512, "sigma": tf.nn.tanh},
            ], row_norm=False,)
            learned_graph = k.build_kernel_matrix(feature_vec)
            # run message passing
            for i, block in enumerate(self.dmps_blocks):
                feature_vec = self._message_passing(
                    feature_vec, block, 
                    learned_graph, layer = i,
                    sigma = tf.nn.relu,
                )
            # pool
            final_vec = tf.reduce_max(feature_vec, axis = 1)
            return final_vec

    def _message_passing(
        self, input, dmps_block, graph, sigma = tf.nn.tanh, layer = 1
    ):
        with tf.variable_scope("message_passing_{}".format(layer)):
            # matmul works on 3d tensors apparently
            diffusion = pool_on_kernel(input, graph, op="max")
            # denoising block
            if dmps_block == DMPSBlock.DENOISE:
                alpha = tf.get_variable(
                    "diffusion_constant", [1], dtype=tf.float32
                )
                diffusion = alpha * input + (1-alpha) * diffusion
            # single nn layer
            dims = tf.shape(diffusion)
            diffusion = row_wise_mlp(
                diffusion, 
                [{"nodes": 256, "sigma": sigma}], name="linear_layer",
                identity_supplement=(dmps_block == DMPSBlock.DENOISE),
                init_pos=True,
            )
            # residual block
            if dmps_block == DMPSBlock.RESIDUAL:
                diffusion = diffusion + input
        return diffusion

if __name__ == '__main__':
    a = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [3, 1, 7]]], dtype=tf.float32)
    final = MessagePassing(a, [DMPSBlock.DMPS] * 3).get_tf_train_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(final))