"""
----
Description:
    Implementation of a 2-d Bessel Convolutional Layer using Tensorflow/Keras.
    This layer can be made E(2)-invariant thanks to the use of the
    Bessel functions of the first kind.
----
           Author: Valentin Delchevalerie
         Creation: 17-12-2021
Last modification: 17-12-2021
----
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
import einops
from getTransMat import getTransMat
from initializers import ConstantTensorInitializer, FourierBesselInitializer, CNNWarmupInitializer

from keras.backend import print_tensor

class BesselConv2d(keras.layers.Layer):
    """
    Main class: define the BesseConv2d layer
    """
    def __init__(self, k, C_out, strides=1, padding='VALID', reflex_inv=False, scale_inv=False, 
                 scales=[-2,0,2], activation=None, TensorCorePad=True, CNNWarmup=None, name=None, **kwargs):
        """
        Initialization of the layer. Called only once, before any training.

        * k is the size of the sliding window used for the convolution
        * C_out is the number of filters in the layer
        * strides is the classic strides parameters used in convolution
        * k_max is the maximum value of k_mj considered for the Bessel
          decomposition of the image. If k_max is set to 'auto', then a 
          value is infered based on k
        * padding is the classic padding parameters used in convolution
        * reflex_inv can be set to True in order to add invariance to reflections
        * activation is the activation function used on the output of the layer
          Available activations are ['relu', 'sigmoid', 'tanh', None]
        * When TensorCorePad is set to True, then padding is used to make mod(m_max+1,8)
          and mod(j_max+1,8) equal to 0. It makes possible for TF to use Tensor Cores.
          Note that this is only possible if mixed precision is also used.
        * CNNWarmup is a string that represent the path to a pretrained CNN model that
          will be used to initialize the weights of the layer. The model should be with
          the same architecture as the one used for the BesselConv2d layer, and with the
          same layer names.
        """

        super(BesselConv2d, self).__init__(name=name, **kwargs)

        if k % 2 == 0:
            ValueError("Kernel size 'k' should be an odd number.")
        if not isinstance(C_out, int) or C_out < 1:
            ValueError("'C_out' should be an integer > 0")
        if not isinstance(strides, int) or strides < 1:
            ValueError("'strides' should be an integer > 0")
        if padding not in ['VALID', 'SAME']:
            ValueError("'padding' should be 'VALID' or 'SAME'")
        if reflex_inv not in [True, False]:
            ValueError("'reflex_inv' should be set to True or False")
        if scale_inv not in [True, False]:
            ValueError("'scale_inv' should be set to True or False")
        if activation not in ['relu', 'sigmoid', 'tanh', None]:
            ValueError("'activation' should be 'relu', 'sigmoid', 'tanh' or None")
        if TensorCorePad not in [True, False]:
            ValueError("'TensorCorePad' should be set to True or False")
        if not isinstance(CNNWarmup, str) or CNNWarmup != None:
            ValueError("'CNNWarmup' should be a string or None")
        
        self.k = k
        self.C_out = C_out
        self.strides = strides
        self.padding = padding
        self.reflex_inv = reflex_inv
        self.scale_inv = scale_inv
        self.scales = scales
        self.activation = activation
        self.TensorCorePad = TensorCorePad
        self.CNNWarmup = CNNWarmup
        
        if self.scale_inv == False:
            self.scales = [0]


    def build(self, input_shape):
        """
        This function builds the model to match a particular input shape.
        It is called only once, when the __call__() method is called for the
        first time. One could also consider calling __build__() directly.

        * input_shape is expected to be in the form (N, H, W, C_in)
        * self.w_r & self.w_i are the real and imaginary parts of the
          filters in the Fourier-Bessel transform domain
        * self.all_T_r & self.all_T_i are the real and imaginary parts 
          of the transformation matrices used to compute the effective filters
        """

        # Get the number of input channels
        self.C_in = input_shape[3]

        # Get the transformation matrices
        k_max = np.pi * (self.k / 2.)
        self.all_T_r = []
        self.all_T_i = []
        for scale in self.scales:

            transMat, self.m_max, self.j_max, Nyquist_m_max, K = getTransMat(self.k+scale, k_max, self.TensorCorePad)

            transMat_r = tf.math.real(transMat)
            transMat_r = tf.reshape(
                tf.transpose(
                    transMat_r,
                    (2,0,1,3)
                ),
                shape=(self.m_max+1, (self.k+scale)**2, self.j_max+1)
            )

            self.all_T_r.append(self.add_weight(
                initializer=ConstantTensorInitializer(transMat_r),
                shape=(self.m_max+1, (self.k+scale)**2, self.j_max+1),
                trainable=False,
                name='transMat_r_'+str(scale))
            )

            transMat_i = tf.math.imag(transMat)
            transMat_i = tf.reshape(
                tf.transpose(
                    transMat_i,
                    (2,0,1,3)
                ),
                shape=(self.m_max+1, (self.k+scale)**2, self.j_max+1)
            )

            self.all_T_i.append(self.add_weight(
                initializer=ConstantTensorInitializer(transMat_i),
                shape=(self.m_max+1, (self.k+scale)**2, self.j_max+1),
                trainable=False,
                name='transMat_i_'+str(scale))
            )
    
        # Initialize trainable weights
        # Tensors has shape (p_max, C_in, C_out)
        if self.CNNWarmup is None:
            
            self.w_r = self.add_weight(
                shape=(self.m_max+1, self.j_max+1, self.C_in * self.C_out),
                initializer=FourierBesselInitializer(K=K, m_max=Nyquist_m_max, imag=False), 
                trainable=True,
                name='weights_real_part'
            )

            self.w_i = self.add_weight(
                shape=(self.m_max+1, self.j_max+1, self.C_in * self.C_out),
                initializer=FourierBesselInitializer(K=K, m_max=Nyquist_m_max, imag=True), 
                trainable=True,
                name='weights_imag_part'
            )

        else:

            transMat, self.m_max, self.j_max, Nyquist_m_max, K = getTransMat(self.k, k_max, self.TensorCorePad)
            w_r, w_i = CNNWarmupInitializer(m_max=self.m_max, 
                                            j_max=self.j_max, 
                                            transMat=transMat, 
                                            path=self.CNNWarmup, 
                                            layer_name=self.name)

            self.w_r = self.add_weight(
                shape=(self.m_max+1, self.j_max+1, self.C_in * self.C_out),
                initializer=ConstantTensorInitializer(w_r), 
                trainable=True,
                name='weights_real_part'
            )

            self.w_i = self.add_weight(
                shape=(self.m_max+1, self.j_max+1, self.C_in * self.C_out),
                initializer=ConstantTensorInitializer(w_i), 
                trainable=True,
                name='weights_imag_part'
            )

        # Remove parameters when k_mj > k_max
        # For m = 0
        n_zeros = (self.j_max+1)*(self.C_in*self.C_out)
        for j in range(1, self.j_max+1):
            if K[0,j] == 0.:
                n_zeros += self.C_in*self.C_out
        # For m > 0
        for m in range(1,self.m_max+1):
            if m > Nyquist_m_max:
                n_zeros += 2*self.C_in*self.C_out*(self.j_max+1)
                continue
            for j in range(1, self.j_max+1):
                if K[m,j] == 0:
                    n_zeros += 2*self.C_in*self.C_out

        # Initialize the biases
        # There are as many biases as the number of filters of the layer (C_out)
        self.b = self.add_weight(
            shape=(self.C_out,),
            initializer=tf.keras.initializers.GlorotNormal,
            trainable=True,
            name='biases'
        )

        self.multiplier = self.add_weight(
            shape=(self.C_out,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.),
            trainable=True,
            name='multiplier'
        )

        self.n_params = 2*(self.m_max+1)*(self.j_max+1)*self.C_in*self.C_out - n_zeros + self.C_out
        
    
    def call(self, inputs):
        """
        This function computes the activation of the layer given (a) particular input(s).
        inputs is of shape (N, W, H, C_in).
        """

        # ----
        # Updating filters using the transformation matrix
        # ----

        all_a = []
        for i, scale in enumerate(self.scales):

            # tf.nn.conv2d only takes 4-d tensors as input.
            # n_out, m_max, real and imag are then wrapped together before performing convolutions.
            # They will be unwrapped later.

            if not self.reflex_inv:

                w_r = tf.math.add(
                    tf.linalg.matmul(self.all_T_r[i], self.w_r),
                    -tf.linalg.matmul(self.all_T_i[i], self.w_i)
                )
                w_i = tf.math.add(
                    tf.linalg.matmul(self.all_T_r[i], self.w_i),
                    tf.linalg.matmul(self.all_T_i[i], self.w_r)
                )

                self.w = einops.rearrange(
                    [w_r, w_i], 
                    'c m (k1 k2) (b1 b2) -> k1 k2 b1 (c m b2)', 
                    k1=self.k+scale, k2=self.k+scale, b1=self.C_in, c=2, m=self.m_max+1, b2=self.C_out
                )

                # ----
                # Computation of the activation.
                # ----

                if self.padding == 'VALID' and scale > 0:
                    pad = scale // 2
                    output = tf.math.square(
                        tf.nn.conv2d(inputs[:,:,:,:], self.w[:,:,:,:], padding=[[0,0],[pad,pad],[pad,pad],[0,0]], strides=self.strides)
                    )
                elif self.padding == 'VALID' and scale < 0:
                    pad = scale // 2
                    output = tf.math.square(
                        tf.nn.conv2d(inputs[:,-pad:pad,-pad:pad,:], self.w[:,:,:,:], padding='VALID', strides=self.strides)
                    )
                else:
                    output = tf.math.square(
                        tf.nn.conv2d(inputs[:,:,:,:], self.w[:,:,:,:], padding=self.padding, strides=self.strides)
                    )

                all_a.append(
                    tf.math.add(
                        tf.math.multiply(
                            einops.reduce(
                                output, 'b w h (c m b1) -> b w h b1', 'sum', 
                                w=output.shape[1], h=output.shape[2], c=2, m=self.m_max+1, b1=self.C_out
                            ),
                            self.multiplier[tf.newaxis,tf.newaxis,tf.newaxis,:]
                        ),
                        self.b[tf.newaxis,tf.newaxis,tf.newaxis,:]
                    )[:,:,:,:,tf.newaxis]
                )

                #print_tensor(all_a[0], 'output')

            else:

                _w_r = tf.linalg.matmul(self.all_T_r[i], self.w_r)
                _w_i = tf.linalg.matmul(self.all_T_i[i], self.w_r)

                w_r = einops.rearrange(
                    [_w_r, _w_i],
                    'c1 m (k1 k2) (b1 b2) -> k1 k2 b1 (c1 m b2)',
                    c1=2, k1=self.k+scale, k2=self.k+scale, b1=self.C_in, m=self.m_max+1, b2=self.C_out
                )

                _w_r = tf.linalg.matmul(self.all_T_r[i], self.w_i)
                _w_i = tf.linalg.matmul(self.all_T_i[i], self.w_i)

                w_i = einops.rearrange(
                    [_w_r, _w_i],
                    'c1 m (k1 k2) (b1 b2) -> k1 k2 b1 (c1 m b2)',
                    c1=2, k1=self.k+scale, k2=self.k+scale, b1=self.C_in, m=self.m_max+1, b2=self.C_out
                )

                self.w = einops.rearrange(
                    [w_r,w_i], 
                    'c2 k1 k2 b1 (c1 m b2) -> k1 k2 b1 (c1 m b2) c2', 
                    c1=2, k1=self.k+scale, k2=self.k+scale, b1=self.C_in, c2=2, m=self.m_max+1, b2=self.C_out
                )

                # ----
                # Computation of the activation.
                # ----

                if self.padding == 'VALID' and scale > 0:
                    pad = scale // 2
                    output = tf.math.add(
                        tf.math.square(
                            tf.nn.conv2d(inputs[:,:,:,:], self.w[:,:,:,:,0], padding=[[0,0],[pad,pad],[pad,pad],[0,0]], strides=self.strides)
                        ),
                        tf.math.square(
                            tf.nn.conv2d(inputs[:,:,:,:], self.w[:,:,:,:,1], padding=[[0,0],[pad,pad],[pad,pad],[0,0]], strides=self.strides)
                        )
                    )
                elif self.padding == 'VALID' and scale < 0:
                    pad = scale // 2
                    output = tf.math.add(
                        tf.math.square(
                            tf.nn.conv2d(inputs[:,-pad:pad,-pad:pad,:], self.w[:,:,:,:,0], padding='VALID', strides=self.strides)
                        ),
                        tf.math.square(
                            tf.nn.conv2d(inputs[:,-pad:pad,-pad:pad,:], self.w[:,:,:,:,1], padding='VALID', strides=self.strides)
                        )
                    )
                else:
                    output = tf.math.add(
                        tf.math.square(
                            tf.nn.conv2d(inputs[:,:,:,:], self.w[:,:,:,:,0], padding=self.padding, strides=self.strides)
                        ),
                        tf.math.square(
                            tf.nn.conv2d(inputs[:,:,:,:], self.w[:,:,:,:,1], padding=self.padding, strides=self.strides)
                        )
                    )

                all_a.append(
                    tf.math.add(
                        einops.reduce(
                            output, 'b w h (c m b1) -> b w h b1', 'sum', 
                            w=output.shape[1], h=output.shape[2], c=2, m=self.m_max+1, b1=self.C_out
                        ),
                        self.b[tf.newaxis,tf.newaxis,tf.newaxis,:]
                    )[:,:,:,:,tf.newaxis]
                )

        a = tf.concat(all_a, axis=-1)

        if self.scale_inv:
            idx = tf.argmax(tf.math.reduce_sum(a, axis=[1,2,3], keepdims=False), axis=-1)
            if a.shape[0] == None:
                # When building the model, tf.gather will not work
                a = a[:,:,:,:,0]
            else:
                a = tf.gather(a, idx, axis=-1, batch_dims=1)
        else:
            a = a[:,:,:,:,0]

        if self.activation == 'relu':
            return tf.keras.activations.relu(a)
        elif self.activation == 'sigmoid':
            return tf.keras.activations.sigmoid(a)
        elif self.activation == 'tanh':
            return tf.keras.activations.tanh(a)
        else:
            return a


    def get_config(self):
        """
        Generates a config in order to save the model at its current state.

        A model using BesselConv2d layer(s) can then be saved using
            >> model.save('./model.h5')
        and loaded with
            >> model = tf.keras.models.load_model('./model.h5', custom_objects={'BesselConv2d': BesselConv2d})
        """

        config = super(BesselConv2d, self).get_config()
        config.update({"k": self.k,
                       "C_out": self.C_out,
                       "strides": self.strides,
                       "padding": self.padding,
                       "reflex_inv": self.reflex_inv,
                       "scale_inv": self.scale_inv,
                       "scales": self.scales,
                       "activation": self.activation})

        return config


    def cast_inputs(self, inputs):
        # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)