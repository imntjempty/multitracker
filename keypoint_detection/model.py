

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 

from multitracker.keypoint_detection import heatmap_drawing

# <network architecture>
from tensorflow.keras.applications.resnet_v2 import preprocess_input

def Encoder(config,inputs):
    # takes input image and encodes it with pretrained resnet model
    #model_name = "ResNet50V2"

    inputs = preprocess_input(inputs)
    net = tf.keras.applications.ResNet152V2(input_tensor=inputs,
                                            include_top=False,
                                            weights='imagenet',
                                            pooling='avg')
    for i,layer in enumerate(net.layers):
        print('layer',layer.name,i,layer.output.shape)
        layer.trainable = False 
    #    if layer.name == "conv4_block6_2_bn":
    #        feature_activation = net[i]
    feature_activation = net.get_layer("max_pooling2d_1")
    model = tf.keras.models.Model(name="ImageNet Encoder",inputs=net.input,outputs=[feature_activation.output])
    return model 

def upsample(nfilters, kernel_size, strides=2, norm_type='batchnorm', act = tf.keras.layers.Activation('relu')):
    initializer = 'he_normal' #tf.random_normal_initializer(0., 0.02)

    if strides == 1:
        func = tf.keras.layers.Conv2D
    else:
        func = tf.keras.layers.Conv2DTranspose

    result = tf.keras.Sequential()
    result.add(
        func(nfilters, kernel_size, strides=strides,
            padding='same',
            kernel_initializer=initializer))

    if norm_type is not None:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(act)

    return result 

def Decoder(config,encoder):
    # takes encoded tensor and produces image sized heatmaps for each keypoint type
    x = encoder.output

    x = upsample(512,3,1)(x)
    fs = 256
    while x.shape[1] < config['img_height']:
        x = upsample(fs,3)(x)
        fs = fs // 2 
    x = upsample(len(config['keypoint_names']),1,1,norm_type=None,act=tf.keras.layers.Activation('tanh'))(x)
    x = 2*(x + 1)
    return x 
# </network architecture>

# <data>

def load_raw_dataset(config):
    def load_im(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        #image = tf.image.resize_with_pad(image,160,160,antialias=True)

        # decompose 
        
        w = config['input_image_shape'][1] // (2 + len(config['keypoint_names'])//3)
        
        comp = tf.concat((image[:,:w,:], image[:,w:2*w,:], image[:,2*w:3*w,:], image[:,3*w:4*w,:] ),axis=2)

        crop = tf.image.random_crop( comp, [config['img_height'],config['img_width'], 3+len(config['keypoint_names'])])
        image = crop[:,:,:3]
        heatmaps = crop[:,:,3:] / 255.
        return image, heatmaps

    file_list = tf.data.Dataset.list_files(os.path.join(config['data_dir'],'*.png'))
    data = file_list.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).repeat().batch(config['batch_size']).prefetch(4*config['batch_size'])#.cache()
    print('[*] loaded images from disk')
    return data 

def create_train_dataset(config):
    # make sure that the heatmaps are 
    if not os.path.isdir(config['data_dir']) or len(glob(os.path.join(config['data_dir'],'*.png')))==0:
        heatmap_drawing.randomly_drop_visualiztions(config['project_id'], dst_dir=config['data_dir'])

    config['input_image_shape'] = cv.imread(glob(os.path.join(config['data_dir'],'*.png'))[0]).shape[:2]
    return config 
# </data>


# <train>
def train(config):
    dataset = load_raw_dataset(config)

    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 3])
    encoder = Encoder(config,inputs)
    print('[*] hidden representation',encoder.outputs[0].get_shape().as_list())
    heatmaps = Decoder(config,encoder)

    optimizer = tf.keras.optimizers.Adam(config['lr'])
    model = tf.keras.models.Model(inputs = encoder.input, outputs = heatmaps) #dataset['train'][0],outputsdataset['train'][1])

    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-')
    checkpoint_path = os.path.expanduser("~/checkpoints/keypoint_tracking/%s" % now)
    vis_directory = os.path.join(checkpoint_path,'vis')
    logdir = os.path.join(checkpoint_path,'logs')
    for _directory in [checkpoint_path,logdir]:
        if not os.path.isdir(_directory):
            os.makedirs(_directory)

    writer = tf.summary.create_file_writer(checkpoint_path)
    def show_tb_images(batch_step):
        with tf.device("cpu:0"):
            with writer.as_default():    
                _global_step = tf.convert_to_tensor(batch_step, dtype=tf.int64)
                tf.summary.image("input",(inputs+1.)/2.,step=batch_step)
                tf.summary.image("heatmap",(heatmaps+1)/2,step=batch_step)
      
    ckpt = tf.train.Checkpoint(model = model,
                            optimizer = optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

    def train_step(inp, y, writer, global_step, should_summarize = False):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # autoencode inputs
            predicted_heatmaps = model(inp, training=True)
            
            # L1 loss
            loss_l1 = tf.reduce_mean(tf.abs(predicted_heatmaps - y)) 
            # ssim loss
            #loss_ssmi = calc_ssim_loss(reconstructed_inp, inp)
            #loss = 0.5 * loss_l1 + 0.5 * loss_ssmi
            loss = loss_l1 

        # gradients
        gradients = tape.gradient(loss,model.trainable_variables)
        # update weights
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))

        # write summary
        if should_summarize:
            with tf.device("cpu:0"):
                with writer.as_default():
                    tf.summary.scalar("l1 loss",loss_l1,step=global_step)
                    #tf.summary.scalar("ssmi loss",loss_ssmi,step=global_step)
                    tf.summary.scalar("loss",loss,step=global_step)
                    def im_summary(name,data):
                        #tf.summary.image(name,(data+1)/2,step=global_step)
                        tf.summary.image(name,data,step=global_step)
                    im_summary('image',inp/256.)
                    im_summary('heatmaps0',tf.concat((y[:,:,:,:3], predicted_heatmaps[:,:,:,:3]),axis=2))
                    im_summary('heatmaps1',tf.concat((y[:,:,:,3:6], predicted_heatmaps[:,:,:,3:6]),axis=2))
                    im_summary('heatmaps2',tf.concat((y[:,:,:,6:], predicted_heatmaps[:,:,:,6:]),axis=2))
                    #im_summary('heatmaps1',predicted_heatmaps[:,:,:,3:6])
                    #im_summary('heatmaps2',predicted_heatmaps[:,:,:,6:])
                    writer.flush()    
    n = 0
    
    ddata = {}
    import pickle 

    for epoch in range(config['epochs']):
        start = time.time()
    
        for x,y in dataset:# </train>

            if n < config['max_steps']:
                should_summarize=n%100==0
                train_step(x, y, writer, n, should_summarize=should_summarize)
                n+=1
            
            
        end = time.time()
        print('[*] epoch %i took %f seconds.'%(epoch,end-start))

    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(config['epochs'], ckpt_save_path))

# </train>

def main():
    config = {'batch_size':32, 'img_height':256,'img_width':256}
    config['epochs'] = 10
    config['max_steps'] = 40000
    config['lr'] = 5e-3

    # train on project 2 
    config['project_id'] = 1 

    config['data_dir'] = os.path.join(os.path.expanduser('~/data/multitracker/projects/%i/data' % config['project_id']))

    from multitracker.be import dbconnection
    db = dbconnection.DatabaseConnection()
    config['keypoint_names'] = db.get_keypoint_names(config['project_id'])

    create_train_dataset(config)
    train(config)


if __name__ == '__main__':
    main()