import tensorflow as tf 
import tensorflow_addons as tfa 

def upsample(nfilters, kernel_size, strides=2, dilation = (1,1), norm_type='batchnorm', act = tf.keras.layers.Activation('relu')):
    initializer = ['he_normal', tf.random_normal_initializer(0., 0.02)][0]

    if strides == 1 or strides < 0:
        strides = abs(strides)
        func = tf.keras.layers.Conv2D
    else:
        func = tf.keras.layers.Conv2DTranspose

    result = tf.keras.Sequential()
    result.add(
        func(nfilters, kernel_size, strides=strides,
            dilation_rate=dilation,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            kernel_initializer=initializer))

    if norm_type is not None:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform"))

    if act is not None:
        result.add(act)

    return result 

def Encoder(config,inputs):
    if config['pretrained_encoder']:
        return EncoderPretrained(config, inputs)
    else:
        return EncoderScratch(config, inputs)

def EncoderScratch(config, inputs, norm_input = True, name = "Scratch Encoder"):
    x = inputs
    if norm_input:
        x = (x-128.)/128. 
    #x = inputs / 255.

    fs = 32
    x = upsample(fs,3,-2)(x)
    
    #while x.shape[1] > 64:
    for _ in range(2):
        fs = min(512, fs * 2)
        d = 1
        x = upsample(fs,1,1)(x)
        xx = upsample(fs,(3,1),1,norm_type=None)(x)
        xx = upsample(fs,(1,3),1,norm_type=None)(xx)
        xx = tf.keras.layers.GaussianNoise(0.35)(xx)
        xx = upsample(fs,(3,1),1,dilation=(d,1),norm_type=None)(xx)
        xx = upsample(fs,(1,3),1,dilation=(1,d),act=None,norm_type=None)(xx)
        xx = tf.keras.layers.Dropout(0.5)(xx)
        #xx = tf.keras.layers.BatchNormalization()(xx)
        xx = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(xx)
        x = x + tf.nn.relu6(xx)
        x = upsample(fs,3,-2)(x)
        
    x = upsample(64,1,1)(x)
    model = tf.keras.models.Model(name=name, inputs=inputs,outputs=[x])
    return model 

def EncoderPretrained(config,inputs):
    if not 'backbone' in config:
        config['backbone'] = 'resnet'
        
    if config['backbone'] == "resnet":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif config['backbone'] == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        
    if config['backbone'] == "resnet":
        net = tf.keras.applications.ResNet152V2
    elif config['backbone'] == 'efficientnet':
        net = tf.keras.applications.EfficientNetB7
    
    inputss = preprocess_input(inputs)
    net = net(input_tensor=inputss,
            include_top=False,
            weights='imagenet',
            pooling='avg')

    for i,layer in enumerate(net.layers):
        #print('layer',layer.name,i,layer.output.shape)
        layer.trainable = False 
    
    if config['backbone'] == "resnet":
        layer_name = ['conv3_block8_1_relu',"conv3_block8_preact_relu","max_pooling2d_1"][1] # 32x32
        layer_name = 'conv2_block3_1_relu' # 64x64x64
    elif config['backbone'] == "efficientnet":
        layer_name = ['conv3_block8_1_relu'][0]
    
    feature_activation = net.get_layer(layer_name)
    model = tf.keras.models.Model(name="ImageNet Encoder",inputs=net.input,outputs=[feature_activation.output])
    return model 


def Decoder(config,encoder):
    decoder = [DecoderErfnet, DecoderErfnetSmall, DecoderDefault][2] # graphics_decoder.GraphicsDecoder
    return decoder(config,encoder)
        
def DecoderErfnet(config, encoder, norm_type = "batchnorm"):
    # Total params: 458,217 Trainable params: 284,649 Non-trainable params: 173,568
    x = encoder.output
    fs = x.shape[-1]
    
    x = upsample(fs,3,1)(x)

    # dilated resnet bottleneck
    for iblock in range(8):
        d = [2,4,8,16][iblock%4]
        xx = upsample(fs,(3,1),1,norm_type=None)(x)
        xx = upsample(fs,(1,3),1,norm_type=None)(xx)
        xx = tf.keras.layers.GaussianNoise(0.35)(xx)
        xx = upsample(fs,(3,1),1,dilation=(d,1),norm_type=None)(xx)
        xx = upsample(fs,(1,3),1,dilation=(1,d),act=None,norm_type=None)(xx)
        xx = tf.keras.layers.Dropout(0.5)(xx)
        if norm_type == "batchnorm":
            xx = tf.keras.layers.BatchNormalization()(xx)
        elif norm_type == "instancenorm":
            xx = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(xx)
        x = x + tf.nn.relu6(xx)

    # upsample
    #while x.shape[1] < config['img_height']:
    for _ in range(2):
        fs = max(32, fs // 2 )
        #x = tf.image.resize(x,[2*x.shape[1],2*x.shape[2]])
        x = upsample(fs,3,norm_type=None)(x)
        #x = tf.keras.layers.GaussianNoise(0.15)(x)
        #x = upsample(fs,3,1,norm_type=None)(x)
        r = tf.keras.layers.Dropout(0.5)(x)
        r = upsample(fs,3,1,act=None,norm_type=None)(r)
        if norm_type == "batchnorm":
            r = tf.keras.layers.BatchNormalization()(r)
        elif norm_type == "instancenorm":
            r = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(r)
        x = x + tf.nn.relu6(r)

    no = len(config['keypoint_names'])+1
    if config['autoencoding']:
        no += 3
    
    x = upsample(32,3,1,norm_type=None)(x)
    act = tf.keras.layers.Activation('softmax')
    x = upsample(no,3,1,norm_type=None,act=act)(x)    
    model = tf.keras.models.Model(inputs = encoder.input, outputs = x, name = "Decoder Erfnet") #dataset['train'][0],outputsdataset['train'][1])
    #model.summary()
    
    return model 

def DecoderErfnetSmall(config, encoder):
    x = encoder.output
    fs = x.shape[-1]
    
    # dilated resnet bottleneck
    for iblock in range(8):
        d = [2,4,8,16][iblock%4]
        xx = upsample(fs,(3,1),1,norm_type=None)(x)
        xx = upsample(fs,(1,3),1,norm_type=None)(xx)
        xx = tf.keras.layers.GaussianNoise(0.2)(xx)
        xx = upsample(fs,(3,1),1,dilation=(d,1),norm_type=None)(x)
        xx = upsample(fs,(1,3),1,dilation=(1,d),act=None,norm_type=None)(xx)
        xx = tf.keras.layers.Dropout(0.5)(xx)
        #xx = tf.keras.layers.BatchNormalization()(xx)
        xx = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True, beta_initializer="random_uniform", gamma_initializer="random_uniform")(xx)
        x = tf.nn.relu6(x + xx)

    x = upsample(1+len(config['keypoint_names']),(3,3),1,norm_type=None,act=tf.keras.layers.Activation('softmax'))(x)
    # upsample
    model = tf.keras.models.Model(inputs = encoder.input, outputs = x, name = "Decoder Erfnet Small") #dataset['train'][0],outputsdataset['train'][1])
    model.summary()
    
    return model 


def DecoderDefault(config, encoder):
    # takes encoded tensor and produces image sized heatmaps for each keypoint type
    x = encoder.output

    fs = max(256, x.shape[-1])
    x = upsample(fs,3,1)(x)
    xx = upsample(fs,7,1, dilation = 2)(x)
    xx = tf.keras.layers.Dropout(0.5)(x)
    xx = upsample(fs,7,1, dilation = 8)(xx)
    x = x + xx 

    while x.shape[1] < config['img_height']:
        fs = max(32, fs // 2 )
        x = upsample(fs,4)(x)
        x = tf.keras.layers.GaussianNoise(0.2)(x)
        r = upsample(fs,3,1)(x)
        r = tf.keras.layers.Dropout(0.5)(r)
        r = upsample(fs,3,1)(r)
        x = x + r
        
    no = len(config['keypoint_names'])+1
    if config['autoencoding']:
        no += 3
    
    act = tf.keras.layers.Activation('softmax')
    x = upsample(no,3,1,norm_type=None,act=act)(x)    
    model = tf.keras.models.Model(inputs = encoder.input, outputs = x, name = "Decoder Vanilla") #dataset['train'][0],outputsdataset['train'][1])
    model.summary()
    
    return model 

