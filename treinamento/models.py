import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras import backend as K

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
