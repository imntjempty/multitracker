import tensorflow as tf 
from multitracker.keypoint_detection.nets import upsample

"""
    implement advanced unet 
    
    inspired from Unet++ https://arxiv.org/pdf/1807.10165.pdf

"""



def get_vanilla_model(config):
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
    
    #x = upsample(nf,1,strides=1)(x)
    #y = upsample(nf,1,strides=1)(x)
    #x = x + y 

    # decoding
    for i_block in range(config['n_blocks']-1,-1,-1):
        #nf = min(1024, bf * 2**i_block)
        nf = layers[i_block].shape[3]
        x = upsample(nf,3,strides=2)(x)
        y = upsample(nf,3,strides=1)(x)
        y = upsample(nf,3,strides=1)(y)
        x = x + y
        # append encoder layer
        e = layers[i_block]
        f = upsample(nf,3,strides=1)(e)
        f = upsample(nf,3,strides=1)(f)
        e = e + f
        #new_size = [inputs.shape[0]/2**i_block,inputs.shape[1]/2**(1+i_block)]
        #e = tf.image.resize(e,new_size)
        x = tf.concat([x,e],axis=3)


    # final classification layer
    x = upsample(1+len(config['keypoint_names']),5,2,norm_type=None,act=tf.keras.layers.Activation('softmax'))(x)     
    
    model = tf.keras.Model(inputs=inputs, outputs=[[x]], name="Unet")
    model.summary()
    return model 

def get_efficient_model(config):
    # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    
    IMG_SIZE = 224
    size = (IMG_SIZE, IMG_SIZE)
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    from tensorflow.keras.applications import EfficientNetB0
    weights='imagenet'
    #weights = None
    
    encoder = EfficientNetB0(include_top=False, weights=weights, drop_connect_rate=0.2,input_tensor=inputs)
    encoder.trainable = True 

    encoded_layer_names = [
        'block1a_activation', # (112,112,32)
        'block3a_expand_activation', # (56,56,144)
        'block4a_expand_activation', # (28,28,240)
        'block5a_expand_activation'#, # (14,14,672)
        #'block6a_activation' # (7,7,1152)
    ]

    encoded_layers = []
    for i , l in enumerate(encoder.layers):
        l.trainable = True
        if l.name in encoded_layer_names:
            x = l.output
            #if l.output.shape[3] > 256:
            #    x = upsample(256,1,strides=1)(x)
            encoded_layers.append(x)
            print('efficient',i, l.output.shape, l.name )
        try:
            #print('efficient',i, [w.shape for w in l.weights], l.name )
            #print('efficient',i, l.output.shape, l.name )
            ''
        except Exception as e :
            print(e)




    #down1 = upsample(256,3,strides=-2)(encoded_layers[-1])
    #down2 = upsample(256,3,strides=-2)(encoded_layers[-1])
    #encoded_layers.append( down1 )

    bf = 64
    x = encoded_layers[-1]
    for _ in range(2):
        x = upsample(512,3,strides=1)(x)

    for i_block in range(len(encoded_layers)-1,-1,-1):
        nf = min(512, bf * 2**i_block)
        #print('decoder',i_block,nf)
        ne = encoded_layers[i_block].shape[3]
        # append encoder layer
        e = encoded_layers[i_block]
        f = upsample(ne,3,strides=1)(e)
        f = tf.keras.layers.Dropout(0.5)(f)
        f = upsample(ne,3,strides=1)(f)
        e = e + f
        #new_size = [inputs.shape[0]/2**i_block,inputs.shape[1]/2**(1+i_block)]
        #e = tf.image.resize(e,new_size)
        x = tf.concat([x,e],axis=3)
        
        x = upsample(nf,3,strides=2)(x)
        y = upsample(nf,3,strides=1)(x)
        y = tf.keras.layers.Dropout(0.5)(y)
        y = upsample(nf,3,strides=1)(y)
        x = x + y
    x = upsample(32,3,strides=1)(x)    
    #x = upsample(64,3,strides=1)(x)

    # final classification layer
    x = upsample(1+len(config['keypoint_names']),1,1,norm_type=None,act=tf.keras.layers.Activation('softmax'))(x)     
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=[[x]], name="Efficient Unet")
    model.summary()
    return model 


def get_model(config):
    return get_efficient_model(config)
    #return get_vanilla_model(config)