import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from keras import layers

class CNN(keras.models.Sequential):
    def __init__(self, img_width, img_height, img_depth, alpha=0.01, activation='relu'):
        super(CNN, self).__init__(name='mlp')

        self.alpha = alpha
        
        self.activation = activation
        #self.input_shape = self.set_input_shape(img_width, img_height, img_depth)
        self.build_model(img_width, img_height, img_depth)
        
    
    def set_input_shape(self, img_width, img_height, img_depth):
        if K.image_data_format()=='channels_first':
            self.my_input_shape = img_depth, img_width,img_height
        elif K.image_data_format()=='channels_last':
            self.my_input_shape = img_width, img_height, img_depth
            
    def add_activation(self):
        if self.activation=='relu':
            self.add(Activation('relu'))
        elif self.activation =='lrelu':
            self.add(LeakyReLU(alpha=self.alpha))
        elif self.activation=='elu':
            self.add(ELU(alpha=self.alpha))
        elif self.activation=='selu':
        	self.add(Activation('selu'))
    
    def build_model():
        pass

class LeNet(CNN):
        
    def build_model(self, img_width, img_height, img_depth):
        if K.image_data_format()=='channels_first':
            input_shape=(img_depth, img_width,img_height)
        elif K.image_data_format()=='channels_last':
            input_shape=(img_width,img_height,img_depth)
            
        self.add(Conv2D(6, (5,5), strides=(1,1), input_shape=input_shape, padding='valid', name='conv_1'))
        self.add_activation()
        self.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))
        self.add(BatchNormalization())

        self.add(Conv2D(16, (5,5), strides=(1,1), padding='valid', name='conv_2'))
        self.add_activation()
        self.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))
        self.add(BatchNormalization())

        self.add(Flatten())

        self.add(Dense(120))
        self.add_activation()
        self.add(BatchNormalization())

        self.add(Dense(84))
        self.add_activation()
        self.add(BatchNormalization())

        self.add(Dense(1))
        self.add(Activation('sigmoid'))

class AlexNet(CNN):
        
    def build_model(self, img_width, img_height, img_depth):
        if K.image_data_format()=='channels_first':
            input_shape=(img_depth, img_width,img_height)
        elif K.image_data_format()=='channels_last':
            input_shape=(img_width,img_height,img_depth)
        
        # 1st Convolutional Layer
        self.add(Conv2D(filters=96, input_shape=(256,256,1), kernel_size=(11,11),\
         strides=(4,4), padding='valid'))
        self.add_activation()
        # Pooling 
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        self.add(BatchNormalization())

        # 2nd Convolutional Layer
        self.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        self.add_activation()
        # Pooling
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        self.add(BatchNormalization())

        # 3rd Convolutional Layer
        self.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.add_activation()
        # Batch Normalisation
        self.add(BatchNormalization())

        # 4th Convolutional Layer
        self.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.add_activation()
        # Batch Normalisation
        self.add(BatchNormalization())

        # 5th Convolutional Layer
        self.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        self.add_activation()
        # Pooling
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        self.add(BatchNormalization())

        # Passing it to a dense layer
        self.add(Flatten())
        # 1st Dense Layer
        self.add(Dense(4096, input_shape=(256*256*3,)))
        self.add_activation()
        # Add Dropout to prevent overfitting
        self.add(Dropout(0.4))
        # Batch Normalisation
        self.add(BatchNormalization())

        # 2nd Dense Layer
        self.add(Dense(4096))
        self.add_activation()
        # Add Dropout
        self.add(Dropout(0.4))
        # Batch Normalisation
        self.add(BatchNormalization())

        # 3rd Dense Layer
        self.add(Dense(1000))
        self.add_activation()
        # Add Dropout
        self.add(Dropout(0.4))
        # Batch Normalisation
        self.add(BatchNormalization())

        # Output Layer
        self.add(Dense(1))
        self.add(Activation('sigmoid'))


def MobileNet(input_shape,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              input_tensor=None,
              pooling=None,
              classes=1000,
              activation='relu'):

    img_input = layers.Input(shape=input_shape)

    #conv1
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(img_input)
    x = layers.Conv2D(32, (3,3),
                      padding='valid',
                      use_bias=False,
                      strides=(2,2),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=-1, name='conv1_bn')(x)
    x = add_activation(x, activation)

    # dw_conv1
    x = _depthwise_conv_block(x, 64, alpha, activation, depth_multiplier, block_id=1)

    # dw_conv2
    x = _depthwise_conv_block(x, 128, alpha, activation, depth_multiplier,
                              strides=(2, 2), block_id=2)

    # dw_conv3
    x = _depthwise_conv_block(x, 128, alpha, activation, depth_multiplier,
                              block_id=3)

    # dw_conv4
    x = _depthwise_conv_block(x, 256, alpha, activation, depth_multiplier,
                              strides=(2, 2), block_id=4)

    # dw_conv5
    x = _depthwise_conv_block(x, 256, alpha, activation, depth_multiplier,
                              block_id=5)

    # dw_conv6
    x = _depthwise_conv_block(x, 512, alpha, activation, depth_multiplier,
                              strides=(2, 2), block_id=6)

    # dw_conv7
    x = _depthwise_conv_block(x, 512, alpha, activation, depth_multiplier, block_id=7)

    # dw_conv8
    x = _depthwise_conv_block(x, 512, alpha, activation, depth_multiplier, block_id=8)

    # dw_conv9
    x = _depthwise_conv_block(x, 512, alpha, activation, depth_multiplier, block_id=9)

    # dw_conv10
    x = _depthwise_conv_block(x, 512, alpha, activation, depth_multiplier, block_id=10)

    # dw_conv11
    x = _depthwise_conv_block(x, 512, alpha, activation, depth_multiplier, block_id=11)

    # dw_conv12
    x = _depthwise_conv_block(x, 1024, alpha, activation, depth_multiplier,
                              strides=(2, 2), block_id=12)

    # dw_conv13
    x = _depthwise_conv_block(x, 1024, alpha, activation, depth_multiplier, block_id=13)

    if K.image_data_format() == 'channels_first':
        shape = (int(1024 * alpha), 1, 1)
    else:
        shape = (1, 1, int(1024 * alpha))

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape(shape, name='reshape_1')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    x = layers.Conv2D(classes, (1, 1),
                      padding='same',
                      name='conv_preds')(x)
    x = layers.Reshape((classes,), name='reshape_2')(x)
    x = layers.Activation('sigmoid', name='act_sigmoid')(x)

    return Model(img_input, x, name='MobileNet')

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, activation, depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)

    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)

    x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)

    x = add_activation(x, activation)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)

    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)

    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def add_activation(inputs, activation):

    if activation=='relu':
        x = layers.ReLU(6.)(inputs)
    elif activation =='lrelu':
        x = layers.LeakyReLU()(inputs)
    elif activation=='elu':
        x = layers.ELU()(inputs)
    elif activation=='selu':
        x = layers.Activation('selu')(inputs)

    return x

