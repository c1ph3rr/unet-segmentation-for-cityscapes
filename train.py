import tensorflow as tf
from glob import glob
import os
import unet
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Tensorflow', tf.__version__)

image_list = sorted(glob('dataset/train_images/*'))
mask_list = sorted(glob('dataset/2_channel_mask/*'))
print('Found {} images \nFound {} masks'.format(len(image_list), len(mask_list)))
H, W = 512, 512
batch_size = 4
lr = 0.0001
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_image(path, mask=False):
    img = tf.io.read_file(path)
    if not mask:
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, size=[H, W])
        img /= 255.
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, size=[H, W]) > 0
        img = tf.cast(img, tf.float32)
    return img

def load_data(image_path, mask_path):
    return get_image(image_path), get_image(mask_path, mask=True)


train_ds = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
train_ds = train_ds.shuffle(256)
train_ds = train_ds.map(load_data, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.repeat()
train_ds = train_ds.prefetch(AUTOTUNE)
print(train_ds)


def dice_loss(pred, actual):
    num = 2 * tf.reduce_sum((pred * actual), axis=-1)
    den = tf.reduce_sum((pred + actual), axis=-1)
    return 1 - (num + 1) / (den + 1)


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = unet.model(512, 512, 1)
    model.compile(optimizer = tf.keras.optimizers.Adam(0.0001), 
                  loss = dice_loss, 
                  metrics = ['accuracy'])

history = model.fit(train_ds, epochs=5, steps_per_epoch=len(image_list)//batch_size, workers=16, use_multiprocessing=True)

model.save_weights('weights.h5')
print('Weights saved')
