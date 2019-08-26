from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization, Input, Conv2DTranspose, concatenate, UpSampling2D, Dropout

def conv_block(inp, filters, name, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal'):
    '''
    Convolutional block with 2 convolutional layers
    '''
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, name='conv'+name+'a')(inp)
    x = BatchNormalization(name='batch'+name+'a')(x)
    x = Activation('relu', name='activation'+name+'a')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, name='conv'+name+'b')(x)
    x = BatchNormalization(name='batch'+name+'b')(x)
    x = Activation('relu', name='activation'+name+'b')(x)
    return x

def pool_block(inp, pool_size, strides, name):
    pool = MaxPool2D(pool_size=pool_size, strides=strides, name='pool'+name)(inp)
    return pool

def upscale_block(inp, skip, filters, name, kernel_size=2, strides=2, kernel_initializer='he_normal'):
    '''
    Upscale block with Conv2dtranspose
    '''
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, name='transpose'+name)(inp)
    x = concatenate([skip, x], name='concate'+name)
    return x

def model(H, W, classes):
    inp = Input((H, W, 3), name='input')

    conv1 = conv_block(inp, 64, '1')
    skip1 = conv1
    conv1 = pool_block(conv1, 2, 2, '1')

    conv2 = conv_block(conv1, 128, '2')
    skip2 = conv2
    conv2 = pool_block(conv2, 2, 2, '2')

    conv3 = conv_block(conv2, 256, '3')
    skip3 = conv3
    conv3 = pool_block(conv3, 2, 2, '3')

    conv4 = conv_block(conv3, 512, '4')
    skip4 = conv4
    conv4 = pool_block(conv4, 2, 2, '4')

    conv5 = conv_block(conv4, 1024, '5')

    conv6 = conv_block(conv5, 512, '6')
    conv6 = upscale_block(conv6, skip4, 512, '1')

    conv7 = conv_block(conv6, 256, '7')
    conv7 = upscale_block(conv7, skip3, 256, '2')

    conv8 = conv_block(conv7, 128, '8')
    conv8 = upscale_block(conv8, skip2, 128, '3')

    conv9 = conv_block(conv8, 64, '9')
    conv9 = upscale_block(conv9, skip1, 64, '4')
    conv9 = conv_block(conv9, 64, '10')

    if classes == 1:
        conv9 = Conv2D(classes, 1, padding='same', activation='sigmoid', kernel_initializer='he_normal', name='conv11')(conv9)
    else:
        conv9 = Conv2D(classes, 1, padding='same', activation='softmax', kernel_initializer='he_normal', name='conv11')(conv9)

    model = Model(inp, conv9, name='unet_'+str(classes))
    model.save('unet_' + str(classes) + '.h5')
    print('Loading UNET')

    return model

# model = model(512, 512, 1)