"""

    python3.7 -m multitracker.keypoint_detection.model --project_id 3
"""

import os
import numpy as np 
import tensorflow as tf 
import tensorflow_addons as tfa
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 

from multitracker.be import dbconnection

db = dbconnection.DatabaseConnection()
from multitracker.keypoint_detection import heatmap_drawing, stacked_hourglass
from multitracker.keypoint_detection.nets import Encoder, Decoder
from multitracker.be import video 



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

mse = tf.keras.losses.MeanSquaredError()
def get_loss(predicted_heatmaps, y, config, mode = "train"):
    if mode == "test":
        return tf.reduce_mean(tfa.losses.SigmoidFocalCrossEntropy(False)(y, predicted_heatmaps))

    if config['loss'] == 'l1':
        loss = tf.reduce_mean(tf.abs(predicted_heatmaps - y))

    elif config['loss'] == 'l2':
        #loss = tf.reduce_mean(tf.nn.l2_loss(predicted_heatmaps - y)) / 5000.
        loss = tf.reduce_mean( mse(y, predicted_heatmaps) )
    
    elif config['loss'] == 'dice':
        a = 2 * tf.reduce_sum(predicted_heatmaps * y, axis=-1 )
        b = tf.reduce_sum(predicted_heatmaps + y, axis=-1 )
        loss = tf.reduce_mean(1 - (a+1)/(b+1))

    elif config['loss'] == 'focal':
        loss_func = tfa.losses.SigmoidFocalCrossEntropy(False)
        if config['autoencoding']:
            loss = loss_func(y[:,:,:,:-3],predicted_heatmaps[:,:,:,:-3])
        else:
            loss = loss_func(y, predicted_heatmaps)
        loss = tf.reduce_mean(loss)

        #loss += tf.reduce_mean(tf.abs(predicted_heatmaps - y))
    elif config['loss'] == 'normed_l1':
        diff = tf.abs(predicted_heatmaps - y)
        diff = diff / tf.reduce_mean(diff)
        loss = tf.reduce_mean(diff)
        
    else:
        raise Exception("Loss function not supported! try l1, normed_l1, l2, focal or dice")
    return loss 

def get_model(config):
    if config['num_hourglass'] == 1:
        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        encoder = Encoder(config,inputs)
        print('[*] hidden representation',encoder.outputs[0].get_shape().as_list())
        model = Decoder(config,encoder)
        return model
    else:
        return stacked_hourglass.get_model(config)

# </network architecture>

# <data>

def load_raw_dataset(config,mode='train', image_directory = None):
    
    if mode=='train' or mode == 'test':
        image_directory = os.path.join(config['data_dir'],'%s' % mode)
    else:
        video_id = db.get_random_project_video(config['project_id'])
        #image_directory = video.get_frames_dir(config['project_id'], video_id)
        image_directory = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'test')
    [H,W,_] = cv.imread(glob(os.path.join(os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'test'),'*.png'))[0]).shape
    [Hcomp,Wcomp,_] = cv.imread(glob(os.path.join(image_directory,'*.png'))[0]).shape
    
    def load_im(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        
        # decompose hstack to dstack
        w = W#config['input_image_shape'][1] // (2 + len(config['keypoint_names'])//3)
        h = H // 2
        
        if mode == 'train' or mode == 'test':
            # now stack depthwise for easy random cropping and other augmentation
            parts = []
            for ii in range(Wcomp//W):#0, 1+len(config['keypoint_names'])//3):
                cc = image[:,ii*W:(ii+1)*W,:]
                parts.append(cc)
            comp = tf.concat(parts,axis=2)
            #comp = tf.concat((image[:,:w,:], image[:,w:2*w,:], image[:,2*w:3*w,:], image[:,3*w:4*w,:] ),axis=2)
            comp = comp[:,:,:(3+len(config['keypoint_names']))]

            if mode == 'train':
                # random scale augmentation
                h += int(np.random.uniform(-h/7,h/7)) #h += int(tf.random.uniform([],-h/7,h/7))

                # apply augmentations
                # random rotation
                if tf.random.uniform([]) > 0.5:
                    random_angle = tf.random.uniform([1],-25.*np.pi/180., 25.*np.pi/180.)  
                    comp = tfa.image.rotate(comp, random_angle)
                
            # add background of heatmap
            background = 255 - tf.reduce_sum(comp[:,:,3:],axis=2)
            comp = tf.concat((comp,tf.expand_dims(background,axis=2)),axis=2)

            # crop
            crop = tf.image.random_crop( comp, [h,h, 1+3+len(config['keypoint_names'])])
            crop = tf.image.resize(crop,[config['img_height'],config['img_width']])
            
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

    #file_list = tf.data.Dataset.list_files(os.path.join(image_directory,'*.png'), shuffle=False)
    file_list = sorted(glob(os.path.join(image_directory,'*.png')))
    # resample list to fill batch
    #rest = len(file_list)%config['batch_size']
    #if rest>0:
    #    file_list = file_list + file_list[:rest]
    file_list_tf = tf.data.Dataset.from_tensor_slices(file_list)
    data = file_list_tf.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['batch_size']).prefetch(4*config['batch_size'])#.cache()
    if mode == 'train':
        data = data.shuffle(512)
    return file_list, data 

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
            if i < int(1+0.8 * len(files)):
                new_f = f.replace(config['data_dir'],config['data_dir']+'/train')
            else:
                new_f = f.replace(config['data_dir'],config['data_dir']+'/test')
            os.rename(f,new_f)

        if 0:
            from multitracker.keypoint_detection import feature_augment
            feature_augment.augment_dataset(config['project_id'])

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
#@tf.function 
def train(config):
    filelist_train, dataset_train = load_raw_dataset(config,'train')
    filelist_test, dataset_test = load_raw_dataset(config,'test')

    model = get_model(config)

    # decaying learning rate 
    decay_steps, decay_rate = 3000, 0.95
    lr = tf.keras.optimizers.schedules.ExponentialDecay([config['lr_scratch'],config['lr']][int(config['pretrained_encoder'])], decay_steps, decay_rate)
    optimizer = tf.keras.optimizers.Adam(lr)
    

     

    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    str_samples = '%i_samples' % len(filelist_train)
    checkpoint_path = os.path.expanduser("~/checkpoints/keypoint_tracking/%s-%s-%iHG-%s" % (config['project_name'],str_samples,config['num_hourglass'],now))
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
            loss = 0. 
            #hourglass_losses = []
            all_predicted_heatmaps = model(inp, training=True)
            for predicted_heatmaps in all_predicted_heatmaps:
                if not predicted_heatmaps.shape[1] == y.shape[1] or predicted_heatmaps.shape[2] == y.shape[2]:
                    predicted_heatmaps = tf.image.resize(predicted_heatmaps, x.shape[1:3])
        
                loss += get_loss(predicted_heatmaps, y, config) / config['num_hourglass']
            
        # clipped gradients
        gradients = tape.gradient(loss,model.trainable_variables)
        gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        
        # update weights
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        

        # write summary
        if should_summarize:
            should_test = should_summarize and global_step % 500 == 0
            if should_test:
                # test data
                test_loss = 0.0
                nt = 0
                for xt,yt in dataset_test:
                    predicted_test = model(xt,training=False)
                    if not predicted_test[0].shape[1] == y.shape[1]:
                        predicted_test = [tf.image.resize(p, x.shape[1:3]) for p in predicted_test]
    
                    if config['autoencoding']:
                        yt = tf.concat((yt,xt),axis=3)
                    test_loss += get_loss(predicted_test[-1], yt, config, 'test')
                    nt += 1 
                test_loss = test_loss / nt 

            with tf.device("cpu:0"):
                def im_summary(name,data):
                    tf.summary.image(name,data,step=global_step)
                with writer_train.as_default():
                    tf.summary.scalar("loss %s" % config['loss'],loss,step=global_step)
                    im_summary('image',inp/255.)
                    for kk in range(1+(y.shape[3]-1)//3):
                        im_summary('heatmaps-%i'%kk,tf.concat((y[:,:,:,kk*3:kk*3+3], predicted_heatmaps[:,:,:,kk*3:kk*3+3]),axis=2))
                    im_summary('background', tf.concat((tf.expand_dims(y[:,:,:,-1],axis=3),tf.expand_dims(predicted_heatmaps[:,:,:,-1],axis=3)),axis=2))
                    
                    tf.summary.scalar("learning rate", lr(global_step), step = global_step)
                    writer_train.flush()    

                if should_test:            
                    with writer_test.as_default():
                        tf.summary.scalar("loss %s" % config['loss'],test_loss,step=global_step)
                        im_summary('image',xt/255.)
                        for i, pre in enumerate(predicted_test): 
                            for kk in range((yt.shape[3]-1)//3):
                                im_summary('heatmaps%i-%i'%(kk,i),tf.concat((yt[:,:,:,:3], pre[:,:,:,:3]),axis=2))
                            im_summary('background-%i'%i, tf.concat((tf.expand_dims(yt[:,:,:,-1],axis=3),tf.expand_dims(pre[:,:,:,-1],axis=3)),axis=2))
                            if config['autoencoding']:
                                im_summary('reconstructed_image',tf.concat((xt/255.,pre[:,:,:,-3:]),axis=2))
                        writer_test.flush()
        return loss

    n = 0
    
    ddata = {}
    import pickle 
    ttrainingstart = time.time()
    epoch = -1
    while epoch < config['epochs'] and time.time()-ttrainingstart<config['max_hours'] * 60. * 60.:
        epoch += 1 
        start = time.time()
        epoch_steps = 0
        epoch_loss = 0.0

        if 1:#try:
            for x,y in dataset_train:# </train>
                # mixup augmentation
                if np.random.random() < 0.9:
                    if config['mixup'] and np.random.random() > 0.5:
                        x, y = mixup(x,y) 
                    else:
                        if config['cutmix'] and np.random.random() > 0.5:
                            x, y = cutmix(x,y)
                    
                #x = tf.keras.layers.experimental.preprocessing.RandomContrast(0.4)(x,training=True)
                #if config['random_contrast'] and np.random.random() > 0.5:
                #(x - mean) * contrast_factor + mean

                if n < config['max_steps']:
                    should_summarize=n%50==0
                    try:
                        epoch_loss += train_step(x, y, writer_train, writer_test, n, should_summarize=should_summarize)
                    except Exception as e:
                        print(e)
                else:
                    return 

                if n % 1000 == 0:
                    ckpt_save_path = ckpt_manager.save()
                    print('[*] saving model to %s'%ckpt_save_path)
                    model.save(os.path.join(checkpoint_path,'trained_model.h5'))
                    if n % 5000 == 0:
                        model.save(os.path.join(checkpoint_path,'trained_model-%ik.h5'%(n//1000)))

                n+=1
                epoch_steps += 1
        #except Exception as e:
        #    print(e)
        epoch_loss = epoch_loss / epoch_steps
        end = time.time()
        print('[*] epoch %i (step %i) took %f seconds with train loss %f.'%(epoch,n, end-start,epoch_loss))

    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(config['epochs'], ckpt_save_path))
    return checkpoint_path

# </train>
def get_config(project_id = 3):
    config = {'batch_size': 8, 'img_height': 256,'img_width': 256}
    config['epochs'] = 1000000
    config['max_steps'] = 150000
    config['max_hours'] = 30.
    config['lr'] = 2e-5 * 5
    config['lr_scratch'] = 1e-4
    config['loss'] = ['l1','dice','focal','normed_l1','l2'][2]
    if config['loss'] == 'l2':
        config['lr'] = 2e-4
    config['autoencoding'] = [False, True][0]
    config['pretrained_encoder'] = [False,True][1]
    config['mixup'] = [False, True][1]
    config['cutmix'] = [False, True][1]
    config['num_hourglass'] = 8

    config['project_id'] = project_id
    config['project_name'] = db.get_project_name(project_id)

    config['data_dir'] = os.path.join(os.path.expanduser('~/data/multitracker/projects/%i/data' % config['project_id']))

    config['keypoint_names'] = db.get_keypoint_names(config['project_id'])

    config['backbone'] = ["resnet","efficientnet"][0]
    return config 

def main(args):
    config = get_config(args.project_id)
    print(config,'\n')
    config['video_id'] = int(args.video_id)
    create_train_dataset(config)
    checkpoint_path = train(config)
    predict.predict(config, checkpoint_path, int(args.project_id), int(args.video_id))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    main(args)

    
    #main(args)