import tensorflow as tf 
from multitracker.keypoint_detection.nets import upsample

"""
    implement advanced unet 
    
    inspired from Unet++ https://arxiv.org/pdf/1807.10165.pdf

"""

def get_model(config):
    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs/127.5-1.
    
    x = upsample(64,5,strides=-2)(x)
    layers = [x]

    bf = 64
    # encoding
    for i_block in range(config['n_blocks']):
        nf = min(1024, bf * 2**i_block)
        x = upsample(nf,3,strides=-2)(x)
        y = upsample(nf,3,strides=1)(x)
        y = upsample(nf,3,strides=1)(y)
        x = x + y
        layers.append(x)
    
    # decoding
    for i_block in range(config['n_blocks']-1,-1,-1):
        nf = min(1024, bf * 2**i_block)
        x = upsample(nf,3,strides=2)(x)
        y = upsample(nf,3,strides=1)(x)
        y = upsample(nf,3,strides=1)(y)
        x = x + y
        # append encoder layer
        e = layers[i_block]
        e = upsample(nf,1,strides=1)(e)
        e = upsample(nf,1,strides=1)(e)
        #new_size = [inputs.shape[0]/2**i_block,inputs.shape[1]/2**(1+i_block)]
        #e = tf.image.resize(e,new_size)
        x = tf.concat([x,e],axis=3)


    # final classification layer
    x = upsample(1+len(config['keypoint_names']),5,2,norm_type=None,act=tf.keras.layers.Activation('softmax'))(x)     
    
    model = tf.keras.Model(inputs=inputs, outputs=[[x]], name="Unet")
    model.summary()
    return model 