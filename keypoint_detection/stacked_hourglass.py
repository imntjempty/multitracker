"""

    Stacked Hourglass

    inspired from Stacked Hourglass Networks for Human Pose Estimation
    https://arxiv.org/pdf/1603.06937.pdf

    l2
    more glasses
    inject image features

"""
import tensorflow as tf 
import tensorflow_addons as tfa
from multitracker.keypoint_detection.nets import upsample, EncoderPretrained, Encoder, Decoder, EncoderScratch

def get_model(config, norm_type = "batchnorm"):
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

def get_model_pretrained(config, norm_type = "batchnorm"):
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

