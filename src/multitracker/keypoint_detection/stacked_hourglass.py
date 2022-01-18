"""

    Stacked Hourglass

    inspired from Stacked Hourglass Networks for Human Pose Estimation
    https://arxiv.org/pdf/1603.06937.pdf

    l2
    more glasses
    in first hourglasss:inject image features

"""
import tensorflow as tf 
import tensorflow_addons as tfa
from multitracker.keypoint_detection.nets import upsample, EncoderPretrained, Encoder, Decoder, EncoderScratch
from multitracker.keypoint_detection.blurpool import BlurPool2D

def BottleneckBlock(inputs, filters, strides=1, downsample=False, name=None):
    inb = inputs
    if downsample:
        inb = upsample(filters,1,-strides)(inb)
    b = upsample(filters//2,1,1)(inb)
    b = upsample(filters//2,3,1)(b)
    b = upsample(filters,1,1,norm_type=None,act=None)(b)
    return inb + b 

def hourglass(config,inputs, level, filters):
    up = BottleneckBlock(inputs, filters)
    up = BottleneckBlock(up, filters)
    if config['kp_blurpool']:
        low = BlurPool2D()(inputs)
    else:
        low = tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2)(inputs)
    low = BottleneckBlock(low, filters)
    if level > 1:
        low = hourglass(config, low, level-1, filters)
    else:
        low = BottleneckBlock(low, filters)
    low = BottleneckBlock(low, filters)
    lowup = tf.keras.layers.UpSampling2D(size=2)(low)
    return up + lowup

def get_model(config):
    if not 'kp_blurpool' in config:
        config['kp_blurpool'] = False 
        
    inputs = tf.keras.layers.Input(shape=(config['img_height'], config['img_width'], 3))
    
    from tensorflow.keras.applications import EfficientNetB1, EfficientNetB6
    encoder = EfficientNetB1(include_top=False, weights='imagenet', drop_connect_rate=0.2,input_tensor=inputs)
    # start training with untrainable base
    encoder.trainable = False 
    for l in encoder.layers:
        l.trainable = False 
    encoding = encoder.get_layer('block3a_expand_activation') # 56x56x144
    
    outputs = [ ]
    filters = 256
    x = upsample(filters,1,1,norm_type=None,act=None)(encoding.output) 
    x = BottleneckBlock(x, filters)
    for i in range(config['kp_num_hourglass']):
        x = hourglass(config, x, 3, filters) 
        x = upsample(filters,1,1)(x)
        y = upsample(1+len(config['keypoint_names']),1,1,norm_type=None,act=tf.keras.layers.Activation('softmax'))(x)  
        ybig = tf.keras.layers.Lambda( lambda image: tf.image.resize(image,(config['img_height'], config['img_width']),method = tf.image.ResizeMethod.BICUBIC))(y)
        outputs.append(ybig) 

        # add initial block again 
        if i < config['kp_num_hourglass']:
            x = upsample(filters,1,1,norm_type=None,act=None)(x) + upsample(filters,1,1,norm_type=None,act=None)(y)
    
    net = tf.keras.Model(inputs,outputs,name="StackedHourglass")
    return encoder, net 


if __name__ == "__main__":
    config = {'img_height': 224, 'img_width': 224, 'kp_num_hourglass': 4, 'keypoint_names': 11*'a'}
    config['kp_blurpool'] = bool(0)
    net = get_model(config)
    net.summary()