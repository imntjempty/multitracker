import tensorflow as tf 
from multitracker.keypoint_detection.nets import upsample

"""
    implement advanced unet 
    
    inspired from Unet++ https://arxiv.org/pdf/1807.10165.pdf

"""

freeze_pretrained_layer = False

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

def preprocess(config, x):
    #return x 
    if 'efficientnet' in config['backbone'] or 'psp' in config['backbone']:
        return tf.keras.applications.efficientnet.preprocess_input(x)
    elif config['backbone'] == 'vgg16':
        return tf.keras.applications.vgg16.preprocess_input(x)


def get_vgg16_model(config):
    from tensorflow.keras.applications import VGG16 
    inputs = tf.keras.layers.Input(shape=(config['img_height'], config['img_width'], 3))
    weights = 'imagenet'
    if 'experiment' in config and config['experiment'] == 'B' and config['should_init_pretrained']==False:
        weights = None 
    encoder = VGG16(include_top=False, weights=weights, input_tensor=inputs)
    #encoder.summary()
    encoder.trainable = False 
        
    encoded_layer_names = [
        'block2_conv2', # (112,112,128)
        'block3_conv3', # ( 56, 56,256)
        'block4_conv3', # ( 28, 28,512)
        'block5_conv3', # ( 14, 14,512)
    ]
    outputs = [get_decoded(config, encoder, encoded_layer_names)]
    model = tf.keras.Model(inputs=encoder.inputs, outputs=[outputs], name="VGG16Unet")
    for l in model.layers:
        if 'block' in l.name:
            l.trainable=False
    
    #model.summary()
    return model 

def get_efficientB0_model(config):
    # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    
    size = (config['img_height'], config['img_width'])
    inputs = tf.keras.layers.Input(shape=(config['img_height'], config['img_width'], 3))
    #x = tf.keras.layers.GaussianNoise(20)(inputs)
    x = inputs 
    from tensorflow.keras.applications import EfficientNetB0
    weights = 'imagenet'
    if 'experiment' in config and config['experiment'] == 'B' and config['should_init_pretrained']==False:
        weights = None 
    
    encoder = EfficientNetB0(include_top=False, weights=weights, drop_connect_rate=0.2,input_tensor=x)
    encoder.trainable = bool(0)#True 
    encoder.summary()
    encoded_layer_names = [
        'block1a_activation', # (112,112,32)
        'block3a_expand_activation', # (56,56,144)
        'block4a_expand_activation', # (28,28,240)
        'block5a_expand_activation'#, # (14,14,672)
        #'block6a_activation' # (7,7,1152)
    ]

    outputs = [get_decoded(config, encoder, encoded_layer_names)]
    model = tf.keras.Model(inputs=encoder.inputs, outputs=[outputs], name="EfficientUnet")
    for l in model.layers:
        if 'block' in l.name:
            l.trainable=False
    #model.summary()
    return model 

def get_efficientB6_model(config):
    # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    
    size = (config['img_height'], config['img_width'])
    inputs = tf.keras.layers.Input(shape=(config['img_height'], config['img_width'], 3))
    #x = tf.keras.layers.GaussianNoise(20)(inputs)
    x = inputs 
    from tensorflow.keras.applications import EfficientNetB6
    weights = 'imagenet'
    if 'experiment' in config and config['experiment'] == 'B' and config['should_init_pretrained']==False:
        weights = None 
    
    encoder = EfficientNetB6(include_top=False, weights=weights, drop_connect_rate=0.2,input_tensor=x)
    encoder.trainable = True 
    encoder.summary() 

    encoded_layer_names = [
        'block1a_activation', # (112,112,32)
        'block3a_expand_activation', # (56,56,144)
        'block4a_expand_activation', # (28,28,240)
        'block5a_expand_activation'#, # (14,14,672)
        #'block6a_activation' # (7,7,1152)
    ]

    outputs = [get_decoded(config, encoder, encoded_layer_names)]
    model = tf.keras.Model(inputs=encoder.inputs, outputs=[outputs], name="LargeEfficientUnet")
    for l in model.layers:
        if 'block' in l.name:
            l.trainable=False
            #print('[*] not training layer %s' % l.name)
    
    #model.summary()
    return model 

def get_decoded(config, encoder, encoded_layer_names):
    encoded_layers = []
    for i , l in enumerate(encoder.layers):
        #l.trainable = True
        if l.name in encoded_layer_names:
            encoded_layers.append(l.output)
            #print('efficient',i, l.output.shape, l.name )
    
    bf = 64
    x = encoded_layers[-1]
    if 0:
        for _ in range(2):
            x = upsample(512,3,strides=1)(x)
            #x = tf.keras.layers.Dropout(0.5)(x)

    for i_block in range(len(encoded_layers)-1,-1,-1):
        nf = min(256, bf * 2**i_block)
        #print('decoder',i_block,nf)
        ne = encoded_layers[i_block].shape[3]
        # append encoder layer
        e = encoded_layers[i_block]
        if 0:
            f = upsample(ne,3,strides=1)(e)
            f = tf.keras.layers.Dropout(0.5)(f)
            f = upsample(ne,3,strides=1)(f)
            e = e + f
        
        #new_size = [inputs.shape[0]/2**i_block,inputs.shape[1]/2**(1+i_block)]
        #e = tf.image.resize(e,new_size)
        x = tf.concat([x,e],axis=3)
        x = upsample(nf,1,strides=1)(x)
        x = upsample(nf,3,strides=2)(x)
        if 1:
            y = upsample(nf,3,strides=1)(x)
            #y = tf.keras.layers.Dropout(0.5)(y)
            y = upsample(nf,3,strides=1)(y)
            x = x + y
    x = upsample(32,3,strides=1)(x)    
    #x = upsample(64,3,strides=1)(x)

    # final classification layer
    x = upsample(1+len(config['keypoint_names']),1,1,norm_type=None,act=tf.keras.layers.Activation('softmax'))(x)     
    return x    

def get_psp_model(config):
    ### https://arxiv.org/abs/1612.01105 
    
    def pyramid_pooling_block(input_tensor, bin_sizes):
        concat_list = [input_tensor]
        w = config['img_width']//8
        h = config['img_height']//8
        #concat_list[0] = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w,h)))(concat_list[0])

        for bin_size in bin_sizes:
            x = tf.keras.layers.AveragePooling2D(pool_size=(h//bin_size, w//bin_size), strides=(h//bin_size, w//bin_size))(input_tensor)
            x = upsample(128, 3, strides=-2)(x)
            x = upsample(128/len(bin_sizes),1, strides=1)(x)
            print('pyramid',bin_size,x.shape)
            x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (h,w)))(x)
            concat_list.append(x)
        return tf.keras.layers.concatenate(concat_list)

    size = (config['img_height'], config['img_width'])
    inputs = tf.keras.layers.Input(shape=(config['img_height'], config['img_width'], 3))
    #x = tf.keras.layers.GaussianNoise(20)(inputs)
    x = inputs 
    from tensorflow.keras.applications import EfficientNetB6
    weights = 'imagenet'
    if 'experiment' in config and config['experiment'] == 'B' and config['should_init_pretrained']==False:
        weights = None 
    
    encoder = EfficientNetB6(include_top=False, weights=weights, drop_connect_rate=0.2,input_tensor=x)
    encoder.trainable = True 
    #encoder.summary() 

    encoded_layer_name = 'block4a_expand_activation' # (28,28,240)
    for i , l in enumerate(encoder.layers):
        if l.name == encoded_layer_name:
            x = l.output
     
    nf = 256 
    x = upsample(nf,1,strides=1)(x)
    # add some extra resnet blocks 
    for i_block in range(16):
        y = upsample(nf,3,strides=1,act=None)(x)
        y = tf.keras.layers.Dropout(0.5)(y)
        y = upsample(nf,3,strides=1)(y)
        x = x + y
    
    x = pyramid_pooling_block(x, [2,4,6,12])
    x = upsample(64,3,strides=1)(x)    
    # final classification layer
    x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (config['img_height'],config['img_width'])))(x)
    x = upsample(1+len(config['keypoint_names']),1,1,norm_type=None,act=tf.keras.layers.Activation('softmax'))(x)     
    
    model = tf.keras.Model(inputs=encoder.inputs, outputs=[[x]], name="PSP")
    for l in model.layers:
        if 'block' in l.name:
            l.trainable=False
            #print('[*] not training layer %s' % l.name)
    
    model.summary()
    return model 


def get_model(config):
    if config['backbone'] == 'efficientnet':
        return get_efficientB0_model(config)
    elif config['backbone'] == 'efficientnetLarge':
        return get_efficientB6_model(config)
    elif config['backbone'] == 'vgg16':
        return get_vgg16_model(config)
    elif config['backbone'] == 'psp':
        return get_psp_model(config)
    #return get_vanilla_model(config)