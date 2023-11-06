import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, concatenate, Conv2DTranspose, Dropout

def fcn32_model(input_shape, num_classes, filters=16, name='FCN32'):
    inputs = Input(shape=input_shape)
    
    # Encoder (VGG-like)
    # Blok 1
    conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Blok 2
    conv2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Blok 3
    conv3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3pred = Conv2D(num_classes, (1,1), activation='linear', padding='same', kernel_initializer=tf.keras.initializers.Zeros())(pool3)


    # Blok 4
    conv4 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4pred = Conv2D(num_classes, (1,1), activation='linear', padding='same', kernel_initializer=tf.keras.initializers.Zeros())(pool4)

    # Blok 5
    conv5 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5pred = Conv2D(num_classes, (1,1), activation='relu', padding='same')(pool5)
    # Selesai Tahap Encoding

    # FCN-32s
    fcn_32_conv = Conv2D(num_classes, (1, 1), activation='softmax')(pool5pred)
    fcn_32_output = UpSampling2D(size=(32, 32), interpolation='bilinear', name='fcn_32')(fcn_32_conv)
    
    model = Model(inputs=inputs, outputs=fcn_32_output , name='FCN32')
    return model

def fcn16_model(input_shape, num_classes, filters=16, name='FCN16'):
    inputs = Input(shape=input_shape)
    
    # Encoder (VGG-like)
    # Blok 1
    conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Blok 2
    conv2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Blok 3
    conv3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3pred = Conv2D(num_classes, (1,1), activation='linear', padding='same', kernel_initializer=tf.keras.initializers.Zeros())(pool3)


    # Blok 4
    conv4 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4pred = Conv2D(num_classes, (1,1), activation='linear', padding='same', kernel_initializer=tf.keras.initializers.Zeros())(pool4)

    # Blok 5
    conv5 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5pred = Conv2D(num_classes, (1,1), activation='relu', padding='same')(pool5)
    # Selesai Tahap Encoding
    
    # FCN-16s
    pool5_upsample = UpSampling2D(size=(2,2), interpolation='bilinear')(pool5pred)
    fcn_16_add = Add()([pool5_upsample, pool4pred])
    fcn_16_conv = Conv2D(num_classes, (1, 1), activation='softmax')(fcn_16_add)
    fcn_16_output = UpSampling2D(size=(16, 16), interpolation='bilinear', name='fcn_16')(fcn_16_conv)
    
    
    model = Model(inputs=inputs, outputs=fcn_16_output, name='FCN16')
    return model

def fcn8_model(input_shape, num_classes, filters=16, name='FCN8'):
    inputs = Input(shape=input_shape)
    
    # Encoder (VGG-like)
    # Blok 1
    conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Blok 2
    conv2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Blok 3
    conv3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(filters*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3pred = Conv2D(num_classes, (1,1), activation='linear', padding='same', kernel_initializer=tf.keras.initializers.Zeros())(pool3)


    # Blok 4
    conv4 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4pred = Conv2D(num_classes, (1,1), activation='linear', padding='same', kernel_initializer=tf.keras.initializers.Zeros())(pool4)

    # Blok 5
    conv5 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(filters*8, (3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5pred = Conv2D(num_classes, (1,1), activation='relu', padding='same')(pool5)
    # Selesai Tahap Encoding
    
    # FCN-16s
    pool5_upsample = UpSampling2D(size=(2,2), interpolation='bilinear')(pool5pred)
    fcn_16_add = Add()([pool5_upsample, pool4pred])
    
    # FCN-8s
    fcn_16_upsample = UpSampling2D(size=(2,2), interpolation='bilinear')(fcn_16_add)
    fcn_8_add = Add()([fcn_16_upsample, pool3pred])
    fcn_8_conv = Conv2D(num_classes, (1, 1), activation='softmax')(fcn_8_add)
    fcn_8_output = UpSampling2D(size=(8, 8), interpolation='bilinear', name='fcn_8')(fcn_8_conv)
    
    model = Model(inputs=inputs, outputs=fcn_8_output, name='FCN8')
    return model

# Unet 
from tensorflow import keras
from tensorflow.keras import layers

def double_conv_block(x, n_filters):

    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = layers.BatchNormalization()(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = layers.BatchNormalization()(x)

    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate 
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x

def build_unet_model(input_shape=(192,256,3), filters=64, num_classes=12):

    # inputs
    inputs = layers.Input(shape=input_shape)

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, filters)
    # 2 - downsample
    f2, p2 = downsample_block(p1, filters*2)
    # 3 - downsample
    f3, p3 = downsample_block(p2, filters*4)
    # 4 - downsample
    f4, p4 = downsample_block(p3, filters*8)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, filters*16)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, filters*8)
    # 7 - upsample
    u7 = upsample_block(u6, f3, filters*4)
    # 8 - upsample
    u8 = upsample_block(u7, f2, filters*2)
    # 9 - upsample
    u9 = upsample_block(u8, f1, filters)

    # outputs
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation = "softmax")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model