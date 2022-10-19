import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from BesselConv2d import BesselConv2d
from GaussianBlur2d import GaussianBlur2d

# Checking available devices
print('CPU available(s):', tf.config.list_physical_devices('CPU'))
print('GPU available(s):', tf.config.list_physical_devices('GPU'))

keras.mixed_precision.set_global_policy('mixed_float16')
#strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

# We will profile on the PCAM dataset
images_folder_train = 'C:/Users/vdelchev/Documents/datasets/PCAM/train'
train_path_labels = 'C:/Users/vdelchev/Documents/datasets/PCAM/train_labels.csv'
#images_folder_train = 'D:/datasets/PCAM/train'
#train_path_labels = 'D:/datasets/PCAM/train_labels.csv'

# Train Images and labels
train_map = pd.read_csv(train_path_labels)
train_map['label'] = train_map['label'].astype(str)

# add .tif
train_map.id = train_map.id.apply(lambda name: name + '.tif')

# Split Train and validation
train, _ = train_test_split(train_map, stratify=train_map.label, test_size=0.9, random_state=42)

# We can't load all data in memory at once, so we use a DataGenerator
#Create instance of ImageDataGenerator Class
image_gen_train = ImageDataGenerator(
                    # Rescale
                    rescale=1/255.,
                    # Rotate 0
                    rotation_range=90,
                    fill_mode='nearest',
                    cval=0.,
                    # Shift pixel values
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    # Flip all image
                    horizontal_flip=True,
                    vertical_flip=True,
                    # Random zoom
                    zoom_range=0.0)

# Custom datagenerator
train_datagen = image_gen_train.flow_from_dataframe(dataframe=train,
                                                    directory=images_folder_train,
                                                    x_col='id',
                                                    y_col='label',
                                                    batch_size=64, #16,32,64...
                                                    seed=1,
                                                    shuffle=True,
                                                    class_mode="binary",
                                                    target_size=(96,96))


#with strategy.scope():
model = keras.models.Sequential()

# Block 1
k = 9 ; n_filters = 16
model.add(BesselConv2d(k=k, C_out=n_filters, reflex_inv=False, scale_inv=False, strides=1, padding='VALID', activation=None, name='Bessel_1'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation(keras.activations.tanh))
k = 9 ; n_filters = 16
model.add(BesselConv2d(k=k, C_out=n_filters, reflex_inv=False, scale_inv=False, strides=1, padding='VALID', activation=None, name='Bessel_2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation(keras.activations.tanh))

# Block 2
#model.add(GaussianBlur2d(sigma=0.6, C_in=n_filters))
model.add(keras.layers.AveragePooling2D((2, 2)))

k = 9 ; n_filters = 24
model.add(BesselConv2d(k=k, C_out=n_filters, reflex_inv=False, scale_inv=False, strides=1, padding='VALID', activation=None, name='Bessel_3'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation(keras.activations.tanh))
k = 7 ; n_filters = 24
model.add(BesselConv2d(k=k, C_out=n_filters, reflex_inv=False, scale_inv=False, strides=1, padding='VALID', activation=None, name='Bessel_4'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation(keras.activations.tanh))

# Block 3
#model.add(GaussianBlur2d(sigma=0.6, C_in=n_filters))
model.add(keras.layers.AveragePooling2D((2, 2)))

k = 7 ; n_filters = 32
model.add(BesselConv2d(k=k, C_out=n_filters, reflex_inv=False, scale_inv=False, strides=1, padding='SAME', activation=None, name='Bessel_7'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation(keras.activations.tanh))
k = 7 ; n_filters = 32
model.add(BesselConv2d(k=k, C_out=n_filters, reflex_inv=False, scale_inv=False, strides=1, padding='SAME', activation=None, name='Bessel_8'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation(keras.activations.tanh))

k = 7 ; n_filters = 32
model.add(BesselConv2d(k=k, C_out=n_filters, reflex_inv=False, scale_inv=False, strides=1, padding='VALID', activation=None, name='Bessel_5'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation(keras.activations.tanh))
k = 7 ; n_filters = 64
model.add(BesselConv2d(k=k, C_out=n_filters, reflex_inv=False, scale_inv=False, strides=1, padding='VALID', activation=None, name='Bessel_6'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation(keras.activations.tanh))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Activation(keras.activations.elu))
model.add(keras.layers.Dense(1))

model.build(input_shape=(None, 96, 96, 3))
model.summary()

#
# Training loop
#

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    #clipvalue=5,
    #clipnorm=2.0,
    name="Adam"
)

loss_fcn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(optimizer=optimizer,
                loss=loss_fcn,
                run_eagerly=False,
                jit_compile=True,
                metrics=['accuracy'])

epochs = 1
patience = 3
min_lr = 0.00001

tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', 
                                          histogram_freq=10, 
                                          write_graph=True, 
                                          write_images=False, 
                                          update_freq='epoch', 
                                          profile_batch='150, 250', 
                                          embeddings_freq=10, 
                                          embeddings_metadata=None)

print(model.layers[0].w_r[:,:,0])

history = model.fit(train_datagen,
                    epochs=epochs,
                    batch_size=64,
                    callbacks=[tb_callback])

print(model.layers[0].w_r[:,:,0])