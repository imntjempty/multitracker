"""
    semisupervised gan with cyclegan

    have a cyclegan translating between video frames and semantic segmentation masks
    use unpaired images and learn via cycle consistency loss 
    use paired images sometimes and learn with cross entropy

    translate Frames (="X") and Semantic Segmentation Masks (="Y")
    based on cyclegan https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb 
    based on semisupervised semantic segmentation https://openaccess.thecvf.com/content_ICCV_2017/papers/Souly__Semi_Supervised_ICCV_2017_paper.pdf

"""

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
downsample, upsample = pix2pix.downsample, pix2pix.upsample

import os
import time
import matplotlib.pyplot as plt
from datetime import datetime 
from IPython.display import clear_output
from multitracker.keypoint_detection import model , unet, heatmap_drawing
from multitracker.be import video 
import cv2 as cv 
from glob import glob 

#tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image


def unet_generator(input_channels, output_channels, norm_type='batchnorm'):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
  Returns:
    Generator model
  """
  
  down_stack = [
      downsample(64, 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
      downsample(128, 4, norm_type),  # (bs, 64, 64, 128)
      downsample(256, 4, norm_type),  # (bs, 32, 32, 256)
      downsample(512, 4, norm_type),  # (bs, 16, 16, 512)
      downsample(512, 4, norm_type),  # (bs, 8, 8, 512)
      downsample(512, 4, norm_type),  # (bs, 4, 4, 512)
      downsample(512, 4, norm_type),  # (bs, 2, 2, 512)
      downsample(512, 4, norm_type),  # (bs, 1, 1, 512)
  ]

  up_stack = [
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
      upsample(512, 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
      upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
      upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
      upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
      upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 4, strides=2,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None, None, input_channels])
  x = inputs
  
  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def discriminator(input_channels, norm_type='batchnorm', target=False):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.
  Returns:
    Discriminator model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, input_channels], name='input_image')
  x = inp

  if target:
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
  down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  if target:
    return tf.keras.Model(inputs=[inp, tar], outputs=last)
  else:
    return tf.keras.Model(inputs=inp, outputs=last)



def discriminator_loss(real, generated):
  real_loss = bce(tf.ones_like(real), real)

  generated_loss = bce(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return bce(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

def load_frames(config):
    image_directory = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'train')
    [H,W,_] = cv.imread(glob(os.path.join(image_directory ,'*.png'))[0]).shape
    h = int(H * config['fov'])
    w = int(W * config['fov'])

    def load_im(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        
        crop = tf.image.random_crop( image, [h,h, 3])
        crop = tf.image.resize(crop,[config['img_height'],config['img_width']])
        return crop 

    file_list = sorted(glob(os.path.join(image_directory,'*.png')))
    file_list_tf = tf.data.Dataset.from_tensor_slices(file_list)
    data = file_list_tf.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['batch_size']).prefetch(4*config['batch_size'])#.cache()
    data = data.shuffle(512)
    return data 

def load_segmentations(config,n_masks = 10):
    image_directory = os.path.join(config['data_dir'],'train')
    [H,W,_] = cv.imread(glob(os.path.join(image_directory ,'*.png'))[0]).shape
    h = int(H * config['fov'])
    w = int(W * config['fov'])

    segmentation_directory = '/tmp/segmentations'
    if not os.path.isdir(segmentation_directory): os.makedirs(segmentation_directory)

    if len(glob(os.path.join(segmentation_directory+'/train','*.png'))) == 0:
        # create random masks
        for i in range(n_masks):
            heatmap_drawing.randomly_drop_visualiztions(config['project_id'], dst_dir=segmentation_directory,max_height=config['max_height'],random_maps=True)
     
    return model.load_raw_dataset(config,mode='train', image_directory = segmentation_directory+'/train')[1]

def train(config):

    filelist_train, dataset_sv = model.load_raw_dataset(config,'train')
    iterator = iter(dataset_sv.repeat())

    dataset_frames = load_frames(config)
    dataset_segmentations = load_segmentations(config)

    generator_g = unet.get_model(config) # outputs: keypoints + background
    generator_f = unet_generator(1+len(config['keypoint_names']), 3, norm_type='batchnorm') # outputs RGB frame

    discriminator_x = discriminator(3, norm_type='batchnorm', target=False)
    discriminator_y = discriminator(1+len(config['keypoint_names']), norm_type='batchnorm', target=False)

    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    checkpoint_path = os.path.expanduser("~/checkpoints/keypoint_cyclegan/%s" % now )

    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                            generator_f=generator_f,
                            discriminator_x=discriminator_x,
                            discriminator_y=discriminator_y,
                            generator_g_optimizer=generator_g_optimizer,
                            generator_f_optimizer=generator_f_optimizer,
                            discriminator_x_optimizer=discriminator_x_optimizer,
                            discriminator_y_optimizer=discriminator_y_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    writer_train = tf.summary.create_file_writer(checkpoint_path+'/train')
    
    EPOCHS = 400

    @tf.function
    def uv_train_step(real_x, real_y, step, writer_train):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # Generator G translates X -> Y
            # Generator F translates Y -> X.

            fake_y = generator_g(real_x, training=True)[0]
            cycled_x = generator_f(fake_y, training=True)

            fake_x = generator_f(real_y, training=True)
            cycled_y = generator_g(fake_x, training=True)[0]

            # same_x and same_y are used for identity loss.
            #same_x = generator_f(real_x, training=True)
            #same_y = generator_g(real_y, training=True)

            disc_real_x = discriminator_x(real_x, training=True)
            disc_real_y = discriminator_y(real_y, training=True)

            disc_fake_x = discriminator_x(fake_x, training=True)
            disc_fake_y = discriminator_y(fake_y, training=True)

            # calculate the loss
            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle_loss = (calc_cycle_loss(real_x/127.5-1., cycled_x) + calc_cycle_loss(real_y, cycled_y))/10.

            # Total generator loss = adversarial loss + cycle loss
            total_gen_g_loss = gen_g_loss + total_cycle_loss # + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss #+ identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        # Calculate the gradients for generator and discriminator
        generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
        generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

        if step % 100 == 0:
            with tf.device("cpu:0"):
                with writer_train.as_default():
                    tf.summary.image('real frame',real_x/256.,step=step)
                    tf.summary.image('real map',real_y[:,:,:,:3],step=step)
                    tf.summary.image('fake frame',fake_x,step=step)
                    tf.summary.image('fake map',fake_y[:,:,:,:3],step=step)
                    tf.summary.image('cycled frame',cycled_x,step=step)
                    tf.summary.image('cycled map',cycled_y[:,:,:,:3],step=step)

                    tf.summary.scalar("loss G" , gen_g_loss,step=step)
                    tf.summary.scalar("loss F" , gen_f_loss,step=step)
                    tf.summary.scalar('loss cycle',total_cycle_loss,step=step)
                    tf.summary.scalar("loss Dx" , disc_x_loss,step=step)
                    tf.summary.scalar("loss Dy" , disc_y_loss,step=step)
                    writer_train.flush()

    @tf.function
    def sv_train_step(step, writer_train):
        paired_x, paired_y = iterator.get_next()

        with tf.GradientTape(persistent=True) as tape:
            predicted = generator_g(paired_x, training=True)[0]
            sv_loss = cce(paired_y, predicted)
        generator_g_gradients_sv = tape.gradient(sv_loss, generator_g.trainable_variables)
        generator_g_optimizer.apply_gradients(zip(generator_g_gradients_sv, generator_g.trainable_variables))

        
        
        if step % 100 == 0:
            with tf.device("cpu:0"):
                with writer_train.as_default():
                    tf.summary.scalar('loss focal', sv_loss, step=step)
                    tf.summary.image('sv frame', paired_x/256.,step=step)
                    tf.summary.image('sv gt', tf.concat([paired_y[:,:,:,:3],paired_y[:,:,:,3:6]],axis=2),step=step)
                    tf.summary.image('sv predicted', tf.concat([predicted[:,:,:,:3],predicted[:,:,:,3:6]],axis=2),step=step)
                    writer_train.flush()

    n = 0 
    sv_kickstart = 2000
    for epoch in range(EPOCHS):
        start = time.time()

        
        for image_x, image_y in tf.data.Dataset.zip((dataset_frames, dataset_segmentations)):
            image_y = image_y[1]
            if 1:#try:
                #print(n,'image_x',image_x.shape,image_y.shape)
                tstep = tf.convert_to_tensor(n, dtype=tf.int64)
                #if n < sv_kickstart or tf.random.uniform([]) > 0.5:
                sv_train_step(tstep, writer_train)
                if n > sv_kickstart:
                    uv_train_step(image_x, image_y,tstep , writer_train)


            #except:
            #    pass 
            if n % 10 == 0:
                print ('.', end='')
        
            n+=1

        clear_output(wait=True)
        # Using a consistent image (sample_horse) so that the progress of the model
        # is clearly visible.
        #generate_images(generator_g, sample_horse)

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))

def main(args):
    config = model.get_config(args.project_id)
    config['video_id'] = int(args.video_id)
    print(config,'\n')
    model.create_train_dataset(config)
    checkpoint_path = train(config)
    #predict.predict(config, checkpoint_path, int(args.project_id), int(args.video_id))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    main(args)

    