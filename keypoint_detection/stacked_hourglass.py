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
    
    from tensorflow.keras.applications import EfficientNetB6
    encoder = EfficientNetB6(include_top=False, weights='imagenet', drop_connect_rate=0.2,input_tensor=inputs)
    encoder.trainable = True 
    encoding = encoder.get_layer('block3a_expand_activation') # 56x56x144
    
    outputs = [ ]
    filters = 256
    x = upsample(filters,1,1,norm_type=None,act=None)(encoding.output) 
    x = BottleneckBlock(x, filters)
    for i in range(config['num_hourglass']):
        x = hourglass(config, x, 3, filters) 
        x = upsample(filters,1,1)(x)
        y = upsample(1+len(config['keypoint_names']),1,1,norm_type=None,act=tf.keras.layers.Activation('softmax'))(x)  
        ybig = tf.keras.layers.Lambda( lambda image: tf.image.resize(image,(config['img_height'], config['img_width']),method = tf.image.ResizeMethod.BICUBIC))(y)
        outputs.append(ybig) 

        # add initial block again 
        if i < config['num_hourglass']:
            x = upsample(filters,1,1,norm_type=None,act=None)(x) + upsample(filters,1,1,norm_type=None,act=None)(y)
    
    model = tf.keras.Model(inputs,outputs,name="StackedHourglass")
    return model 


def get_erfnet_model(config, norm_type = "batchnorm"):
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    outputs = [] 
    x = inputs/256.
    # downsample to /4 res
    image_features = upsample(32,3,strides=-2)(x)
    image_features = upsample(64,3,strides=-2)(image_features)

    should_inject_image_features = bool(1)
    fs = 512
    x = upsample(fs,3,strides=1)(image_features) 
    # 8 resblocks for each hourglass stage
    for istage in range(config['num_hourglass']):
        xin = x
        if should_inject_image_features and istage > 0:
            z = tf.concat([image_features,tf.nn.avg_pool2d(outputs[-1],4,4,'SAME')],axis=3)
            z = upsample(fs,1,1)(z)
            x = x + z 
        for iblock in range(8):
            d = [2,4,8,16][iblock%4]
            xx = upsample(fs//2,1,1)(x)
            xx = upsample(fs//2,(3,1),1,norm_type=None)(xx)
            xx = upsample(fs//2,(1,3),1)(xx)
            #xx = tf.keras.layers.GaussianNoise(0.7)(xx)
            xx = upsample(fs//2,(3,1),1,dilation=(d,1),norm_type=None)(xx)
            xx = upsample(fs//2,(1,3),1,dilation=(1,d))(xx)
            xx = tf.keras.layers.Dropout(0.5)(xx)
            xx = upsample(fs,1,1)(xx)
            #xx = tf.keras.layers.BatchNormalization()(xx)
            if norm_type == "batchnorm":
                xx = tf.keras.layers.BatchNormalization()(xx)
            elif norm_type == "instancenorm":
                xx = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(xx)
        x = x + tf.nn.relu6(xx)
        x = x + xin         
        # output (intermediate) heatmap of sidearm in full res
        y = upsample(32,3,2)(x)#up1(x)
        y = upsample(1+len(config['keypoint_names']),3,2,norm_type=None,act=tf.keras.layers.Activation('softmax'))(y) # up2(y)    
        outputs.append( y )
    #outputs.append(  )


    model = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = "ErfnetHourglass")
    #model = tf.keras.models.Model(inputs = inputs, outputs = [decoder0.outputs[0],decoder1.outputs[0]], name = "Decoder Erfnet Hourglass") #dataset['train'][0],outputsdataset['train'][1])
    model.summary()
 
    return model

def get_model_erfnet_pretrained(config, norm_type = "batchnorm"):
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    outputs = [] 
    
    encoder = EncoderPretrained(config,inputs)
    
    # reuse 2x2xupsampling at each stage
    #up1 = upsample(32,3,2,norm_type=None)
    #up2 = upsample(1+len(config['keypoint_names']),3,2,norm_type=None,act=tf.keras.layers.Activation('softmax'))
    should_inject_image_features = True
    fs = 256
    xfin = upsample(fs,3,1)(encoder.outputs[0])
    x = xfin 
    # 8 resblocks for each hourglass stage
    for istage in range(config['num_hourglass']):
        if should_inject_image_features and istage > 0:
            #x = tf.concat([x,encoder.outputs[0]],axis=3)
            x = tf.concat([x,encoder.outputs[0],tf.nn.avg_pool2d(outputs[-1],4,4,'SAME')],axis=3)
            x = upsample(fs,1,1)(x)
        xin = x
        for iblock in range(8):
            d = [2,4,8,16][iblock%4]
            xx = upsample(fs//2,1,1)(x)
            xx = upsample(fs//2,(3,1),1,norm_type=None)(xx)
            xx = upsample(fs//2,(1,3),1,norm_type=None)(xx)
            xx = tf.keras.layers.GaussianNoise(0.7)(xx)
            xx = upsample(fs//2,(3,1),1,dilation=(d,1),norm_type=None)(xx)
            xx = upsample(fs//2,(1,3),1,dilation=(1,d),act=None,norm_type=None)(xx)
            xx = tf.keras.layers.Dropout(0.5)(xx)
            xx = upsample(fs,1,1)(xx)
            #xx = tf.keras.layers.BatchNormalization()(xx)
            if norm_type == "batchnorm":
                xx = tf.keras.layers.BatchNormalization()(xx)
            elif norm_type == "instancenorm":
                xx = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(xx)
        x = x + tf.nn.relu6(xx)
        x = x + xin         
        # output (intermediate) heatmap of sidearm
        y = upsample(32,4,2,norm_type=None)(x)#up1(x)
        y2 = upsample(1+len(config['keypoint_names']),4,2,norm_type=None,act=tf.keras.layers.Activation('softmax'))(y) # up2(y)    
        outputs.append( y2 )
    #outputs.append(  )


    model = tf.keras.models.Model(inputs = encoder.inputs, outputs = outputs, name = "Erfnet Hourglass")
    #model = tf.keras.models.Model(inputs = inputs, outputs = [decoder0.outputs[0],decoder1.outputs[0]], name = "Decoder Erfnet Hourglass") #dataset['train'][0],outputsdataset['train'][1])
    model.summary()
 
    return model

if __name__ == "__main__":
    config = {'img_height': 224, 'img_width': 224, 'num_hourglass': 4, 'keypoint_names': 11*'a'}
    config['kp_blurpool'] = bool(0)
    model = get_model(config)
    model.summary()