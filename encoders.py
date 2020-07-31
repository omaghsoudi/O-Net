import keras
from keras.models import *
from keras.layers import *
import keras.backend as K


################################################################################ Simple
################################################################################
################################################################################
def simple_encoder(input_shape, Channel_order):
	kernel = (3, 3)
	filter_size = 64
	pad = (1, 1)
	pool_size = (2, 2)

	img_input = Input(input_shape)

	x = (ZeroPadding2D(pad, data_format=Channel_order))(img_input)
	x = (Conv2D(filter_size, kernel, data_format=Channel_order, padding='valid'))(x)
	x = (BatchNormalization())(x)
	x = (Activation('relu'))(x)
	x = (MaxPooling2D(pool_size, data_format=Channel_order))(x)
	B1 = x

	x = (ZeroPadding2D(pad, data_format=Channel_order))(x)
	x = (Conv2D(filter_size*2, kernel, data_format=Channel_order , padding='valid'))(x)
	x = (BatchNormalization())(x)
	x = (Activation('relu'))(x)
	x = (MaxPooling2D(pool_size, data_format=Channel_order))(x)
	B2 = x

	x = (ZeroPadding2D(pad, data_format=Channel_order ))(x)
	x = (Conv2D(filter_size*4, kernel, data_format=Channel_order, padding='valid'))(x)
	x = (BatchNormalization())(x)
	x = (Activation('relu'))(x)
	x = (MaxPooling2D(pool_size, data_format=Channel_order))(x)
	B3 = x

	x = (ZeroPadding2D(pad, data_format=Channel_order ))(x)
	x = (Conv2D(filter_size*8, kernel, data_format=Channel_order, padding='valid'))(x)
	x = (BatchNormalization())(x)
	x = (Activation('relu'))(x)
	x = (MaxPooling2D(pool_size, data_format=Channel_order))(x)
	B4 = x

	x = (ZeroPadding2D(pad, data_format=Channel_order ))(x)
	x = (Conv2D(filter_size*8, kernel, data_format=Channel_order, padding='valid'))(x)
	x = (BatchNormalization())(x)
	x = (Activation('relu'))(x)
	x = (MaxPooling2D(pool_size, data_format=Channel_order))(x)
	B5 = x

	return img_input, [B1, B2, B3, B4, B5]



################################################################################ VGG
################################################################################
################################################################################
def vgg_encoder(input_shape, Channel_order):
	# the images should be a multiply of 32
	assert input_shape[0]%32 == 0
	assert input_shape[1]%32 == 0

	kernel = (3, 3)
	filter_size = 64
	pool_size = (2, 2)
	stride = (2, 2)

	img_input = Input(input_shape)

	x = Conv2D(filter_size, kernel, activation='relu', padding='same', name='block1_conv1', data_format=Channel_order )(img_input)
	x = Conv2D(filter_size, kernel, activation='relu', padding='same', name='block1_conv2', data_format=Channel_order )(x)
	x = MaxPooling2D(pool_size, strides=stride, name='block1_pool', data_format=Channel_order)(x)
	B1 = x
	# Block 2
	x = Conv2D(filter_size*2, kernel, activation='relu', padding='same', name='block2_conv1', data_format=Channel_order )(x)
	x = Conv2D(filter_size*2, kernel, activation='relu', padding='same', name='block2_conv2', data_format=Channel_order )(x)
	x = MaxPooling2D(pool_size, strides=stride, name='block2_pool', data_format=Channel_order)(x)
	B2 = x

	# Block 3
	x = Conv2D(filter_size*4, kernel, activation='relu', padding='same', name='block3_conv1', data_format=Channel_order)(x)
	x = Conv2D(filter_size*4, kernel, activation='relu', padding='same', name='block3_conv2', data_format=Channel_order)(x)
	x = Conv2D(filter_size*4, kernel, activation='relu', padding='same', name='block3_conv3', data_format=Channel_order)(x)
	x = MaxPooling2D(pool_size, strides=stride, name='block3_pool', data_format=Channel_order)(x)
	B3 = x

	# Block 4
	x = Conv2D(filter_size*8, kernel, activation='relu', padding='same', name='block4_conv1', data_format=Channel_order)(x)
	x = Conv2D(filter_size*8, kernel, activation='relu', padding='same', name='block4_conv2', data_format=Channel_order)(x)
	x = Conv2D(filter_size*8, kernel, activation='relu', padding='same', name='block4_conv3', data_format=Channel_order)(x)
	x = MaxPooling2D(pool_size, strides=stride, name='block4_pool', data_format=Channel_order)(x)
	B4 = x

	# Block 5
	x = Conv2D(filter_size*8, kernel, activation='relu', padding='same', name='block5_conv1', data_format=Channel_order)(x)
	x = Conv2D(filter_size*8, kernel, activation='relu', padding='same', name='block5_conv2', data_format=Channel_order)(x)
	x = Conv2D(filter_size*8, kernel, activation='relu', padding='same', name='block5_conv3', data_format=Channel_order)(x)
	x = MaxPooling2D(pool_size, strides=kernel, name='block5_pool', data_format=Channel_order)(x)
	B5 = x

	return img_input, [B1, B2, B3, B4, B5]



################################################################################ RESNET
################################################################################
################################################################################
def one_side_pad(x, Channel_order):
	x = ZeroPadding2D((1, 1), data_format=Channel_order)(x)
	x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
	return x

def identity_block(input_tensor, kernel_size, filters, Channel_order, stage, block):
	bn_axis = 3

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters[0], (1, 1) , data_format=Channel_order , name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters[1], kernel_size , data_format=Channel_order ,
	           padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters[2] , (1, 1), data_format=Channel_order , name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = add([x, input_tensor])
	x = Activation('relu')(x)
	return x


def conv_block(input_tensor, Channel_order, kernel_size, filters, stage, block, strides=(2, 2)):
	bn_axis = 3

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters[0], (1, 1), data_format=Channel_order, strides=strides,
	           name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters[1], kernel_size, data_format=Channel_order, padding='same',
	           name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters[2], (1, 1), data_format=Channel_order, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Conv2D(filters[2], (1, 1) , data_format=Channel_order  , strides=strides,
	                  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = add([x, shortcut])
	x = Activation('relu')(x)
	return x


def resnet_encoder(input_shape, Channel_order, input_tensor=None,
	include_top=True, pooling=None, classes=3):

	assert input_shape[0]%32 == 0
	assert input_shape[1]%32 == 0

	img_input = Input(input_shape)
	bn_axis = 3

	x = ZeroPadding2D((3, 3), data_format=Channel_order)(img_input)
	x = Conv2D(64, (7, 7), data_format=Channel_order, strides=(2, 2), name='conv1')(x)
	B1 = x
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3) , data_format=Channel_order , strides=(2, 2))(x)


	x = conv_block(x, Channel_order, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], Channel_order, stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], Channel_order, stage=2, block='c')
	B2 = one_side_pad(x, Channel_order)


	x = conv_block(x, Channel_order, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], Channel_order, stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], Channel_order, stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], Channel_order, stage=3, block='d')
	B3 = x

	x = conv_block(x, Channel_order, 3, [256, 256, 1024], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 1024], Channel_order, stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 1024], Channel_order, stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 1024], Channel_order, stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 1024], Channel_order, stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 1024], Channel_order, stage=4, block='f')
	B4 = x

	x = conv_block(x, Channel_order, 3, [512, 512, 2048], stage=5, block='a')
	x = identity_block(x, 3, [512, 512, 2048], Channel_order, stage=5, block='b')
	x = identity_block(x, 3, [512, 512, 2048], Channel_order, stage=5, block='c')
	B5 = x

	x = AveragePooling2D((7, 7), data_format=Channel_order, name='avg_pool')(x)

	return img_input, [B1, B2, B3, B4, B5]
