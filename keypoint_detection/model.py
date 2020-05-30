

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 

from multitracker.be import dbconnection

db = dbconnection.DatabaseConnection()
from multitracker.keypoint_detection import heatmap_drawing    
from multitracker.be import video 


from tensorflow.keras.applications.resnet_v2 import preprocess_input

def Encoder(config,inputs):
    if config['pretrained_encoder']:
        return EncoderPretrained(config, inputs)
    else:
        return EncoderScratch(config, inputs)

def EncoderScratch(config, inputs):
    x = (inputs-128.)/128. 
    #x = inputs / 255.

    fs = 32
    x = upsample(fs,3,-2)(x)
    
    while x.shape[1] > 16:
        fs = min(512, fs * 2)
        x = upsample(fs,3,1)(x)
        x = upsample(fs,4,-2)(x)
    x = upsample(fs,1,1)(x)
    model = tf.keras.models.Model(name="Scratch Encoder", inputs=inputs,outputs=[x])
    return model 

def EncoderPretrained(config,inputs):
    inputss = preprocess_input(inputs)
    net = tf.keras.applications.ResNet152V2(input_tensor=inputss,
                                            include_top=False,
                                            weights='imagenet',
                                            pooling='avg')
    for i,layer in enumerate(net.layers):
        # print('layer',layer.name,i,layer.output.shape)
        layer.trainable = False 
    layer_name = ['conv3_block8_1_relu',"conv3_block8_preact_relu","max_pooling2d_1"][0]
    feature_activation = net.get_layer(layer_name)
    model = tf.keras.models.Model(name="ImageNet Encoder",inputs=net.input,outputs=[feature_activation.output])
    return model 

def upsample(nfilters, kernel_size, strides=2, norm_type='batchnorm', act = tf.keras.layers.Activation('relu')):
    initializer = ['he_normal', tf.random_normal_initializer(0., 0.02)][1]

    if strides == 1 or strides < 0:
        strides = abs(strides)
        func = tf.keras.layers.Conv2D
    else:
        func = tf.keras.layers.Conv2DTranspose

    result = tf.keras.Sequential()
    result.add(
        func(nfilters, kernel_size, strides=strides,
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            kernel_initializer=initializer))

    if norm_type is not None:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    if act is not None:
        result.add(act)

    return result 

def Decoder(config,encoder):
    # takes encoded tensor and produces image sized heatmaps for each keypoint type
    x = encoder.output

    fs = x.shape[-1]
    x = upsample(fs,7,1)(x)
    
    x = tf.keras.layers.Dropout(0.5)(x)

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
    x = upsample(no,1,1,norm_type=None,act=act)(x)    
    return x 


def calc_ssim_loss(x, y):
  """Computes a differentiable structured image similarity measure."""
  c1 = 0.01**2
  c2 = 0.03**2
  mu_x = tf.nn.avg_pool2d(x, 3, 1, 'VALID')
  mu_y = tf.nn.avg_pool2d(y, 3, 1, 'VALID')
  sigma_x = tf.nn.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
  sigma_y = tf.nn.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
  sigma_xy = tf.nn.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
  ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
  ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
  ssim = ssim_n / ssim_d
  ssim = tf.clip_by_value((1 - ssim) / 2, 0, 1)
  ssim = tf.reduce_mean(ssim)
  return ssim

def get_loss(predicted_heatmaps, y, config):
    if config['loss'] == 'l1':
        loss = tf.reduce_mean(tf.abs(predicted_heatmaps - y))
    elif config['loss'] == 'dice':
        a = 2 * tf.reduce_sum(predicted_heatmaps * y, axis=-1 )
        b = tf.reduce_sum(predicted_heatmaps + y, axis=-1 )
        loss = tf.reduce_mean(1 - (a+1)/(b+1))

        #loss += tf.reduce_mean(tf.norm(tf.reduce_mean(y,axis=[1,2]) ) - tf.reduce_mean(predicted_heatmaps,axis=[1,2]))
    elif config['loss'] == 'recall':
        loss = (1+tf.reduce_mean(y - predicted_heatmaps*y,axis=-1)) / (1+tf.reduce_mean(y+1e-5,axis=-1)) # penalize uncovered y
        loss = tf.reduce_mean(loss)
        loss += tf.reduce_mean(tf.abs(predicted_heatmaps - y)) / 10. # abit of l1 

    elif config['loss'] == 'focal':
        import tensorflow_addons as tfa
        loss_func = tfa.losses.SigmoidFocalCrossEntropy(False)
        if config['autoencoding']:
            loss = loss_func(y[:,:,:,:-3],predicted_heatmaps[:,:,:,:-3])
        else:
            loss = loss_func(y, predicted_heatmaps)
        loss = tf.reduce_mean(loss)
    else:
        raise Exception("Loss function not supported! try l1, focal or dice")
    return loss 

# </network architecture>

# <data>

def load_raw_dataset(config,mode='train', image_directory = None):
    
    if mode=='train' or mode == 'test':
        image_directory = os.path.join(config['data_dir'],'%s' % mode)
    else:
        video_id = db.get_random_project_video(config['project_id'])
        #image_directory = video.get_frames_dir(config['project_id'], video_id)
        image_directory = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), video_id),'test')
    [h,w,_] = cv.imread(glob(os.path.join(image_directory,'*.png'))[0]).shape

    def load_im(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        
        if mode == 'train' or mode == 'test':
            # decompose hstack to dstack
            w = config['input_image_shape'][1] // (2 + len(config['keypoint_names'])//3)
            h = config['input_image_shape'][0]

            # now stack depthwise for easy random cropping and other augmentation
            comp = tf.concat((image[:,:w,:], image[:,w:2*w,:], image[:,2*w:3*w,:], image[:,3*w:4*w,:] ),axis=2)
            comp = comp[:,:,:(3+len(config['keypoint_names']))]

            # add background of heatmap
            background = 255 - tf.reduce_max(comp[:,:,3:],axis=2)
            comp = tf.concat((comp,tf.expand_dims(background,axis=2)),axis=2)
            
            # apply augmentations
            # resize
            #H, W = config['input_image_shape'][:2]
            if 1:
                scalex = np.random.uniform(1.,1.5)
                scaley = np.random.uniform(1.,1.5)
                comp = tf.image.resize(comp, size = (int(scaley*h),int(scalex*w)))
            
                # crop
                crop = tf.image.random_crop( comp, [h,h, 1+3+len(config['keypoint_names'])])
                crop = tf.image.resize(comp,[config['img_height'],config['img_width']])
                
            # split stack into images and heatmaps
            image = crop[:,:,:3]
            heatmaps = crop[:,:,3:] / 255.
            return image, heatmaps
        else:
            ratio = max(config['img_height']/h,config['img_width']/w)
            image = tf.image.resize(image,[int(ratio*xs[1]),int(ratio*xs[2])])
            crop = image[h//2-config['img_height']//2:h//2+config['img_height']//2,w//2-config['img_width']//2:w//2+config['img_width']//2,:]
            print('resized',xs,'to',xdown.shape,'and cropped to',crop.shape)
            
            #h,w,_ = image.shape
            #crop = image[h//2-config['img_height']//2:h//2+config['img_height']//2,w//2-config['img_width']//2:w//2+config['img_width']//2,:]
            #crop = tf.image.resize(crop,[config['img_height'],config['img_width']])
            
            return crop

    file_list = tf.data.Dataset.list_files(os.path.join(image_directory,'*.png'), shuffle=False)
    data = file_list.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['batch_size']).prefetch(4*config['batch_size'])#.cache()
    if mode == 'train':
        data = data.shuffle(1024)
    return data 

def create_train_dataset(config):
    # make sure that the heatmaps are 
    if not os.path.isdir(config['data_dir']) or len(glob(os.path.join(config['data_dir'],'train/*.png')))==0:
        heatmap_drawing.randomly_drop_visualiztions(config['project_id'], dst_dir=config['data_dir'])
        for mode in ['train','test']:
            mode_dir = os.path.join(config['data_dir'],mode)
            if not os.path.isdir(mode_dir):
                os.makedirs(mode_dir)

        files = sorted(glob(os.path.join(config['data_dir'],'*.png')))
        for i, f in enumerate(files):
            # split train test frames
            if i < int(1+0.9 * len(files)):
                new_f = f.replace(config['data_dir'],config['data_dir']+'/train')
            else:
                new_f = f.replace(config['data_dir'],config['data_dir']+'/test')
            os.rename(f,new_f)

    config['input_image_shape'] = cv.imread(glob(os.path.join(config['data_dir'],'train/*.png'))[0]).shape[:2]
    return config 

def mixup(x,y):
    rr = tf.random.uniform(shape=[x.shape[0]],minval=0.0,maxval=0.3)
    rrr = tf.reshape(rr,(x.shape[0],1,1,1))
    xx = x[::-1,:,:,:]
    yy = y[::-1,:,:,:]
    x = rrr * x + (1. - rrr) * xx 
    y = rrr * y + (1. - rrr) * yy
    return x,y 

def cutmix(x,y):
    # https://arxiv.org/pdf/1906.01916.pdf
    # fixing the area of the rectangle to half that of the image, while varying the aspect ratio and position
    s = x.shape
    randw = int(s[2]*np.random.uniform(0.7,0.9))
    randh = int(s[1]*s[2]//2 / randw )
    px = int(np.random.uniform(0,s[2]-randw))
    py = int(np.random.uniform(0,s[1]-randh))
    
    Mx = np.zeros(s)
    My = np.zeros(y.shape)
    Mx[:,py:py+randh,px:px+randw,:] = 1.0 
    My[:,py:py+randh,px:px+randw,:] = 1.0 

    xx = Mx*x + (1.-Mx)*x[::-1,:,:,:] 
    yy = My*y + (1.-My)*y[::-1,:,:,:] 
    return xx, yy
# </data>



# <train>
def train(config):
    dataset_train = load_raw_dataset(config,'train')
    dataset_test = load_raw_dataset(config,'test')

    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 3])
    encoder = Encoder(config,inputs)
    print('[*] hidden representation',encoder.outputs[0].get_shape().as_list())
    heatmaps = Decoder(config,encoder)

    # decaying learning rate 
    decay_steps, decay_rate = 2000, 0.95
    lr = tf.keras.optimizers.schedules.ExponentialDecay(config['lr'], decay_steps, decay_rate)
    optimizer = tf.keras.optimizers.Adam(lr)
    optimizer_ae = tf.keras.optimizers.Adam(config['lr'])

    model = tf.keras.models.Model(inputs = encoder.input, outputs = heatmaps) #dataset['train'][0],outputsdataset['train'][1])
    model.summary() 

    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    checkpoint_path = os.path.expanduser("~/checkpoints/keypoint_tracking/%s" % now)
    vis_directory = os.path.join(checkpoint_path,'vis')
    logdir = os.path.join(checkpoint_path,'logs')
    for _directory in [checkpoint_path,logdir]:
        if not os.path.isdir(_directory):
            os.makedirs(_directory)

    writer_train = tf.summary.create_file_writer(checkpoint_path+'/train')
    writer_test = tf.summary.create_file_writer(checkpoint_path+'/test')
      
    ckpt = tf.train.Checkpoint(model = model, optimizer = optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

    def train_step(inp, y, writer_train, writer_test, global_step, should_summarize = False):
        if y is None:
            s = inp.shape
            y = tf.zeros((s[0],s[1],s[2],1+len(config['keypoint_names'])))
        if config['autoencoding']:
            y = tf.concat((y,inp),axis=3)

        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            predicted_heatmaps = model(inp, training=True)
            loss = get_loss(predicted_heatmaps, y, config)
            #loss_ae = get_loss(predicted_heatmaps[:,:,:,-3:], y[:,:,:,-3:], config)

        # gradients
        gradients = tape.gradient(loss,model.trainable_variables)
        #gradients_ae = tape.gradient(loss_ae,model.trainable_variables)
        # update weights
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        #optimizer_ae.apply_gradients(zip(gradients_ae,model.trainable_variables))

        # write summary
        if should_summarize:
            should_test = should_summarize and global_step % 100 == 0
            if should_test:
                # test data
                test_loss = 0.0
                nt = 0
                for xt,yt in dataset_test:
                    predicted_test = model(xt,training=False)
                    if config['autoencoding']:
                        yt = tf.concat((yt,xt),axis=3)
                    test_loss += get_loss(predicted_test, yt, config)
                    nt += 1 
                test_loss = test_loss / nt 

            with tf.device("cpu:0"):
                def im_summary(name,data):
                    tf.summary.image(name,data,step=global_step)
                with writer_train.as_default():
                    tf.summary.scalar("loss %s" % config['loss'],loss,step=global_step)
                    im_summary('image',inp/255.)
                    im_summary('heatmaps0',tf.concat((y[:,:,:,:3], predicted_heatmaps[:,:,:,:3]),axis=2))
                    im_summary('heatmaps1',tf.concat((y[:,:,:,3:6], predicted_heatmaps[:,:,:,3:6]),axis=2))
                    im_summary('heatmaps2',tf.concat((y[:,:,:,6:8], predicted_heatmaps[:,:,:,6:8]),axis=2))
                    im_summary('background', tf.concat((tf.expand_dims(y[:,:,:,-1],axis=3),tf.expand_dims(predicted_heatmaps[:,:,:,-1],axis=3)),axis=2))
                    
                    if config['autoencoding']:
                        im_summary('reconstructed_image',tf.concat((inp/255.,predicted_heatmaps[:,:,:,-3:]),axis=2))

                    tf.summary.scalar("learning rate", lr(global_step), step = global_step)
                    writer_train.flush()    

                if should_test:            
                    with writer_test.as_default():
                        tf.summary.scalar("loss %s" % config['loss'],test_loss,step=global_step)
                        im_summary('image',xt/255.)
                        im_summary('heatmaps0',tf.concat((yt[:,:,:,:3], predicted_test[:,:,:,:3]),axis=2))
                        im_summary('heatmaps1',tf.concat((yt[:,:,:,3:6], predicted_test[:,:,:,3:6]),axis=2))
                        im_summary('background', tf.concat((tf.expand_dims(yt[:,:,:,-1],axis=3),tf.expand_dims(predicted_test[:,:,:,-1],axis=3)),axis=2))
                        if config['autoencoding']:
                            im_summary('reconstructed_image',tf.concat((xt/255.,predicted_test[:,:,:,-3:]),axis=2))
                        writer_test.flush()

    n = 0
    
    ddata = {}
    import pickle 

    for epoch in range(config['epochs']):
        start = time.time()
    
        for x,y in dataset_train:# </train>
            # mixup augmentation
            if config['mixup']:
                x, y = mixup(x,y) 

            if config['cutmix']:
                x, y = cutmix(x,y)
            #x = tf.keras.layers.experimental.preprocessing.RandomContrast(0.4)(x,training=True)


            if n < config['max_steps']:
                should_summarize=n%50==0
                train_step(x, y, writer_train, writer_test, n, should_summarize=should_summarize)
                
            if n % 1000 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('[*] saving model to %s'%ckpt_save_path)
                model.save(os.path.join(checkpoint_path,'trained_model.h5'))

            n+=1
        end = time.time()
        print('[*] epoch %i took %f seconds.'%(epoch,end-start))

    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(config['epochs'], ckpt_save_path))

# </train>
def get_config():
    config = {'batch_size': 16, 'img_height': 256,'img_width': 256}
    config['epochs'] = 1000000
    config['max_steps'] = 400000
    config['lr'] = 2e-5 * 4
    config['loss'] = ['l1','dice','recall','focal'][3]
    config['autoencoding'] = [False, True][0]
    config['pretrained_encoder'] = True
    config['mixup'] = [False, True][0]
    config['cutmix'] = [False, True][1]

    # train on project 2 
    config['project_id'] = 2

    config['data_dir'] = os.path.join(os.path.expanduser('~/data/multitracker/projects/%i/data' % config['project_id']))

    config['keypoint_names'] = db.get_keypoint_names(config['project_id'])
    return config 

def main():
    config = get_config()
    create_train_dataset(config)
    train(config)


if __name__ == '__main__':
    main()