"""
----
Description:
    Implementation of a 2-d Gaussian blur filter to avoid aliasing
    when using B-CNNs or similar models.
----
           Author: Valentin Delchevalerie
         Creation: 10-11-2021
Last modification: 10-11-2021
----
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras


class GaussianBlur2d(keras.layers.Layer):
    def __init__(self, sigma, C_in=1, name=None, **kwargs):

        super(GaussianBlur2d, self).__init__(name=name, **kwargs)

        self.sigma = sigma
        k = 2*int(round(3*sigma))+1
        self.C_in = C_in

        self.w = tf.Variable(
            initial_value=tf.cast(self._gaussian_kernel(k, self.sigma, C_in, dtype=tf.float32), tf.float16),
            shape=(k, k, self.C_in, 1),
            trainable=False,
            name='weights'
        )
        
    
    def _gaussian_kernel(self, kernel_size, sigma, n_channels, dtype):
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
        g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
        g_norm2d = tf.pow(tf.reduce_sum(g), 2)
        g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


    def call(self, inputs):
        a = tf.nn.depthwise_conv2d(inputs, self.w, [1,1,1,1], 'SAME')
        return a


    def get_config(self):
        config = super(GaussianBlur2d, self).get_config()
        config.update({"sigma": self.sigma,
                       "C_in": self.C_in})

        return config


    def cast_inputs(self, inputs):
        # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)