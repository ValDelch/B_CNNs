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
    def __init__(self, K, m_max, imag=False):
        self.K = K
        self.m_max = m_max
        self.imag = imag

    def __call__(self, shape, dtype=None):
        initializer = tf.keras.initializers.GlorotNormal()
        w = tf.Variable(initializer(shape, dtype=dtype))

        # Remove parameters when k_mj > k_max
        if not self.imag:
            # For m = 0
            for j in range(1, self.K.shape[1]):
                if self.K[0,j] == 0.:
                    w = w[0,j,:].assign(0.)
            # For m > 0
            for m in range(1, self.K.shape[0]):
                if m > self.m_max:
                    w = w[m,:,:].assign(0.)
                    continue
                for j in range(1, self.K.shape[1]):
                    if self.K[m,j] == 0:
                        w = w[m,j,:].assign(0.)
        else:
            # For m = 0
            w = w[0,:,:].assign(0.)
            # For m > 0
            for m in range(1, self.K.shape[0]):
                if m > self.m_max:
                    w = w[m,:,:].assign(0.)
                    continue
                for j in range(1, self.K.shape[1]):
                    if self.K[m,j] == 0:
                        w = w[m,j,:].assign(0.)
           
        return w

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def CNNWarmupInitializer(m_max, j_max, transMat, path=None, layer_name=None):

    model = tf.keras.models.load_model(path, custom_objects={'root_mean_squared_error': root_mean_squared_error})

    for layers in model.layers:
        if layers.name == layer_name:

            # Getting the weights from the CNN layer
            weights = layers.get_weights()
            k = weights[0].shape[0] ; C_in = weights[0].shape[2] ; C_out = weights[0].shape[3]
            weights = weights[0].reshape(k, k, C_in * C_out)

            # Converting them to Fourier-Bessel coefficients
            w_r = np.zeros((m_max+1, j_max+1, C_in*C_out))
            w_i = np.zeros((m_max+1, j_max+1, C_in*C_out))

            transMat = transMat.reshape((k**2, (m_max+1)*(j_max+1)))
            transMat = np.matmul(np.linalg.pinv(np.matmul(np.conj(transMat.T), transMat)), np.conj(transMat.T))
            for i in range(C_in*C_out):
                w = np.matmul(transMat, weights[:,:,i].reshape((k**2,1))).reshape((m_max+1,j_max+1))
                w_r[:,:,i] = np.real(w)
                w_i[:,:,i] = np.imag(w)

            return (tf.convert_to_tensor(w_r, dtype=tf.float32), tf.convert_to_tensor(w_i, dtype=tf.float32))