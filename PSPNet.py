from math import ceil
from keras.models import *
from keras.layers import *

# encoders
from encoders import simple_encoder, vgg_encoder, resnet_encoder



################################################################################ PSPNET
def Interp(x, shape):
	from keras.backend import tf as ktf
	new_height, new_width = shape
	resized = ktf.image.resize_images(x, [new_height, new_width],align_corners=True)
	return resized



def interp_block(prev_layer, level, feature_map_shape):
	kernel = (level, level)
	strides = (level, level)


	prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
	prev_layer = Conv2D(512, (1, 1), strides=(1, 1), use_bias=False)(prev_layer)
	prev_layer = BatchNormalization(momentum=0.95, epsilon=1e-5)(prev_layer)
	prev_layer = Activation('relu')(prev_layer)
	prev_layer = Lambda(Interp, arguments={'shape': feature_map_shape})(prev_layer)

	return prev_layer



def build_pyramid_pooling_module(encoder_out, input_shape):
	"""Build the Pyramid Pooling Module."""

	feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape[:-1])

	interp_block1 = interp_block(encoder_out, 6, feature_map_size)
	interp_block2 = interp_block(encoder_out, 3, feature_map_size)
	interp_block3 = interp_block(encoder_out, 2, feature_map_size)
	interp_block6 = interp_block(encoder_out, 1, feature_map_size)


	encoder_out = Lambda(Interp, arguments={'shape': feature_map_size})(encoder_out)

	out = Concatenate()([encoder_out,
	                     interp_block6,
	                     interp_block3,
	                     interp_block2,
	                     interp_block1])
	return out


def PSPNet(n_classes, input_shape, encoder, activation="softmax"):
	Channel_order = 'channels_last'

	if encoder=="resnet_encoder":
	    img_input, Blocks = resnet_encoder(input_shape, Channel_order)
	elif encoder=="vgg_encoder":
	    img_input, Blocks = vgg_encoder(input_shape, Channel_order)
	elif encoder=="simple_encoder":
	    img_input, Blocks = simple_encoder(input_shape, Channel_order)

	[B1, B2, B3, B4, B5] = Blocks

	x = build_pyramid_pooling_module(B5, input_shape)

	x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
	x = BatchNormalization(momentum=0.95, epsilon=1e-5)(x)
	x = Activation('relu')(x)
	x = Dropout(0.1)(x)

	x = Conv2D(n_classes, (1, 1), strides=(1, 1), name="conv6")(x)
	x = Lambda(Interp, arguments={'shape': (input_shape[0], input_shape[1])})(x)
	x = Activation(activation)(x)

	model = Model(inputs=img_input, outputs=x)

	return model
