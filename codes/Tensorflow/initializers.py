import tensorflow as tf
from keras import backend as K
import numpy as np



class ConstantTensorInitializer(tf.keras.initializers.Initializer):
    """
    Used to initialize weights with a constant tensor
    """
    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self, shape, dtype=None):
        if self.tensor.shape != shape:
            raise ValueError('Wrong shape for the constant tensor.')
        return self.tensor

class FourierBesselInitializer(tf.keras.initializers.Initializer):
    """
    A wrapper for the GlorotNormal initializer that takes k_max into account
    by setting to zero the weights corresponding to k_mj > k_max
    """
    def __init__(self, MASK, C_in, C_out, k, imag=False):
        self.MASK = MASK
        self.C_in = C_in
        self.C_out = C_out
        self.k = k
        self.imag = imag

    def __call__(self, shape, dtype=None):
        fan_in = self.C_in * (self.k**2)
        fan_out = self.C_out * (self.k**2)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(2. / (fan_in)))
        w = tf.Variable(initializer(shape, dtype=dtype))

        # Remove parameters when k_mj > k_max
        if not self.imag:
            for m in range(self.MASK.shape[0]):
                for j in range(self.MASK.shape[1]):
                    if not self.MASK[m,j]:
                        # Value should not be used
                        for _i in range(w.shape[-1]):
                            w = w[m,j,_i].assign(0.)
                        continue

                    #if m == 0 and j == 0:
                    #    w = w[m,j,:].assign(w[m,j,:] * 30.)
                    #elif m == 0 and j == 1:
                    #    w = w[m,j,:].assign(w[m,j,:] * 10.)
                    #else:
                    #    w = w[m,j,:].assign(w[m,j,:] / (1+(m+j)))

        else:
            # For m = 0, no imaginary part
            for _i in range(w.shape[1]):
                for _j in range(w.shape[2]):
                    w = w[0,_i,_j].assign(0.)
            # For m > 0
            for m in range(1, self.MASK.shape[0]):
                for j in range(self.MASK.shape[1]):
                    if not self.MASK[m,j]:
                        # Value should not be used
                        for _i in range(w.shape[-1]):
                            w = w[m,j,_i].assign(0.)
                        continue

                    #w = w[m,j,:].assign(w[m,j,:] / (1+(m+j)))
           
        return w

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))