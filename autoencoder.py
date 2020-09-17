"""

    Denoising Convolutional Autoencoder on single cropped faces
    hidden representation is later used for unsupervised cluster finding
"""

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from datetime import datetime

from multitracker.be import dbconnection

n_latentcode = 256


def upsample_transpconv(filters, size, norm_type='batchnorm', apply_norm=True, apply_dropout=False,activation=tf.nn.relu):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())


  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.Activation(activation))
  
  #result.add(tf.keras.layers.ReLU())

  return result

def downsample_stridedconv(filters, size, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.
  Conv2D => Batchnorm => LeakyRelu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def Encoder(inputs, config={}):
    x = inputs
    # denoising
    x = tf.keras.layers.GaussianNoise(0.2)(x)
    x = downsample_stridedconv(32,(3,3), norm_type='batchnorm', apply_norm=False)(x) # 40
    x = downsample_stridedconv(64,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 20
    x = downsample_stridedconv(64,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 10
    f = x 
    #x = downsample_stridedconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 5
    #x = downsample_stridedconv(256,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 5
    #x = downsample_stridedconv(n_latentcode,(3,3), norm_type='batchnorm', apply_norm=True)(x) 
    return [f,x]

def Decoder(config,encoder):
    x = encoder
    #x = upsample_transpconv(n_latentcode,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 
    #x = llayers.upsample_transpconv(256,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 5
    #x = upsample_transpconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 5
    #x = upsample_transpconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 10
    x = upsample_transpconv(64,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 20
    x = upsample_transpconv(32,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 40
    x = upsample_transpconv(3,(3,3), norm_type='batchnorm', apply_norm=False,activation=tf.tanh)(x) # 80
    return x 

def load_im(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image,channels=3)
    return preprocess(image)

def preprocess(image):
    image = tf.image.resize_with_pad(image,640,640,antialias=True)
    image = tf.cast(image,tf.float32)
    image = (image / 127.5) - 1

    return image

def load_raw_dataset(config):
    image_dir = os.path.join(dbconnection.base_data_dir, 'projects/%i/%i/frames/train' % (config['project_id'],config['video_id']))
    file_list = tf.data.Dataset.list_files(os.path.join(image_dir,'*.png'))
    data = file_list.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).repeat().batch(config['batch_size']).prefetch(4*config['batch_size'])#.cache()
    print('[*] loaded images from disk')
    return file_list, data 
    
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

def train(config=None):
    if config is None:
        config = get_autoencoder_config()
    #symbol_files, dataset = load_single_symbols_dataset(config)
    symbol_files, dataset = load_raw_dataset(config)
    
    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 3])
    feature_extractor, encoder = Encoder(inputs, config)
    print('[*] feature_extractor',feature_extractor.shape,'encoded',encoder.get_shape().as_list())
    reconstructed = Decoder(config,encoder)

    optimizer = tf.keras.optimizers.Adam(config['lr'])
    encoder_model = Model(inputs = inputs, outputs = [feature_extractor,encoder])
    autoencoder = Model(inputs = inputs, outputs = [feature_extractor, reconstructed]) #dataset['train'][0],outputsdataset['train'][1])

    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    checkpoint_path = os.path.expanduser("~/checkpoints/multitracker_ae_bbox/%s/%s" % (config['project_name'], now))
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
                tf.summary.image("reconstructed",(reconstructed+1)/2,step=batch_step)
      
    ckpt = tf.train.Checkpoint(encoder_model=encoder_model)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

    def train_step(inp, writer, global_step, should_summarize = False):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # autoencode inputs
            _, reconstructed_inp = autoencoder(inp, training=True)
            
            # L1 loss
            loss_l1 = tf.reduce_mean(tf.abs(reconstructed_inp - inp)) 
            # ssim loss
            loss_ssmi = calc_ssim_loss(reconstructed_inp, inp)

            loss = 0.5 * loss_l1 + 0.5 * loss_ssmi

        # gradients
        gradients = tape.gradient(loss,autoencoder.trainable_variables)
        # update weights
        optimizer.apply_gradients(zip(gradients,autoencoder.trainable_variables))

        # write summary
        if should_summarize:
            with tf.device("cpu:0"):
                with writer.as_default():
                    tf.summary.scalar("l1 loss",loss_l1,step=global_step)
                    tf.summary.scalar("ssmi loss",loss_ssmi,step=global_step)
                    tf.summary.scalar("loss",loss,step=global_step)
                    def im_summary(name,data):
                        tf.summary.image(name,(data+1)/2,step=global_step)
                    im_summary('image',inp)
                    im_summary('reconstructed',reconstructed_inp)
                    writer.flush()    
    n = 0
    
    ddata = {}
    import pickle 

    for epoch in range(int(1e6)):
        start = time.time()
    
        for inp in dataset:
            _global_step = tf.convert_to_tensor(n, dtype=tf.int64)

            if n < config['max_steps']:
                should_summarize=n%100==0
                train_step(inp,writer,_global_step,should_summarize=should_summarize)
                n+=1
            
            if n == config['max_steps']:
                ckpt_save_path = ckpt_manager.save()
                print('[*] done training, saving checkpoint for step {} at {}'.format(n, ckpt_save_path))
                return ckpt_save_path
               
        end = time.time()
        print('[*] epoch %i took %f seconds.'%(epoch,end-start))
    
def get_autoencoder_config():
    config = {'batch_size':8, 'img_height':640,'img_width':640}
    config['max_steps'] = 15000
    config['lr'] = 1e-4
    return config

if __name__ == '__main__':
    train()