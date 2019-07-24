from set_utils import row_wise_mlp
import tensorflow as tf

class SetModel():
    def get_model(self):
        return tf.squeeze(row_wise_mlp(
            self._get_model(), hidden_sizes=[40], sigma=tf.identity, 
        ))
    
    def _get_model(self):
        raise UnimplementedException

class UnimplementedException(Exception):
    pass