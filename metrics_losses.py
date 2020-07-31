import tensorflow as tf
import keras.backend as K


# A simple loss functions and metrics just for testing

def dice(y_true, y_pred, smooth=K.epsilon()):
    # the first channel is background so it will be ignored
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]
    intersection = K.sum(y_true * y_pred)

    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)



def loss_dice(y_true, y_pred):
    return 1-dice(y_true, y_pred)



def weighting(y_true, y_pred):
    Weights = []
    Y1 = []
    Y2 = []
    for Class in range(K.int_shape(y_pred)[-1]):
        y1 = y_true[:,:,:,Class]
        y2 = y_pred[:,:,:,Class]

        y_true_class1_w = K.sum(y1, axis=(1,2))
        y_true_others = K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2] - y_true_class1_w

        weights = ( y_true_others )/( K.int_shape(y_pred)[1]*K.int_shape(y_pred)[2] )

        weight = weights

        Y1.append(y1)
        Y2.append(y2)
        Weights.append(weight)
    return (Y1, Y2, Weights)



def weighted_dice(y_true, y_pred):
    Y1, Y2, Weights = weighting(y_true, y_pred)

    Sum_weights = 0
    for N, (y1, y2, weight) in enumerate(zip(Y1, Y2, Weights)):
        Sum = K.sum(y1 * y2, axis=(1,2))
        Sum_true = K.sum(y1, axis=(1,2))
        Sum_pred = K.sum(y2, axis=(1,2))
        Sum_weights += weight

        Nominator = weight*2*( K.sum(Sum) )
        Denominator = ( K.sum(Sum_true) + K.sum(Sum_pred)+K.epsilon() )
        result = tf.keras.backend.mean((Nominator/Denominator))

        if N==0:
            DICE = result
        else:
            DICE += result
    return DICE/Sum_weights


def loss_weighted_dice(y_true, y_pred):
    return 1-weighted_dice(y_true, y_pred)
