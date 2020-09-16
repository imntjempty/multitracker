"""
    detect keypoints on cropped animals

"""

import tensorflow as tf
import tensorflow_addons as tfa
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime 
from IPython.display import clear_output
from multitracker.keypoint_detection import model , unet, heatmap_drawing
from multitracker.be import video 
import cv2 as cv 
from glob import glob 
from random import shuffle
import numpy as np 
from datetime import datetime 

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

focal_loss = tfa.losses.SigmoidFocalCrossEntropy(False)
def loss_func(ytrue,ypred):
    return tf.reduce_mean( focal_loss(ytrue,ypred))

def get_roi_crop_dim(project_id, video_id, Htarget):
    """
        we need to crop around the centers of bounding boxes
        the crop dimension should be high enough to see the whole animal but as small as possible
        different videos show animals in different sizes, so we scan the db for all bboxes and take 98% median as size
    """
    [Hframe,Wframe,_] = cv.imread(glob(os.path.join(os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, project_id), video_id),'test'),'*.png'))[0]).shape
    db.execute("select * from bboxes where video_id=%i;" % video_id)
    db_boxxes = [x for x in db.cur.fetchall()]
    assert len(db_boxxes) > 0, "[*] ERROR: no labeled bounding boxes found! please label at least one bounding box for video " + str(video_id) 
        
    deltas = []
    for i, [_, _, frame_idx,x1,y1,x2,y2] in enumerate(db_boxxes):
        deltas.extend([x2-x1,y2-y1])
    deltas = np.array(deltas)
    if 0:
        print('deltas',deltas.shape,deltas.mean(),deltas.std(),'min/max',deltas.min(),deltas.max())
        for pp in [50, 75, 90, 95, 96, 97, 98, 99]:
            print('perc',pp,np.percentile(deltas, pp))
    
    # take 97% percentile with margin of 20%
    crop_dim = np.percentile(deltas, 97) * 1.2
    
    # scale to target frame height
    crop_dim = int(crop_dim * Htarget/Hframe) 

    # not bigger than image 
    crop_dim = min(crop_dim,Htarget)
    return crop_dim
    #return int(Htarget/3.)

def load_roi_dataset(config,mode='train'):
    if mode=='train' or mode == 'test':
        image_directory = os.path.join(config['data_dir'],'%s' % mode)
    [Hframe,Wframe,_] = cv.imread(glob(os.path.join(os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'test'),'*.png'))[0]).shape
    
    [Hcomp,Wcomp,_] = cv.imread(glob(os.path.join(image_directory,'*.png'))[0]).shape
    H = Hcomp 
    w = int(Wframe*Hcomp/Hframe)

    #w = H#int(W*Hcomp/H)
     #config['fov'])
    len_parts = Wcomp // w  
    crop_dim = get_roi_crop_dim(config['project_id'], config['video_id'], Hcomp)
    
    for _mode in ['train','test']:
        if not os.path.isdir(os.path.join(config['roi_dir'],_mode)): os.makedirs(os.path.join(config['roi_dir'],_mode))
        
    if len(glob(os.path.join(config['roi_dir'],'train','*.png')))==0:
        # extract bounding boxes of animals
        # <bboxes>
        frame_bboxes = {}
        db.execute("select * from bboxes where video_id=%i;" % config['video_id'])
        db_boxxes = [x for x in db.cur.fetchall()]
        shuffle(db_boxxes)
        for dbbox in db_boxxes:
            _, _, frame_idx, x1, y1, x2, y2 = dbbox 
            if not frame_idx in frame_bboxes:
                frame_bboxes[frame_idx] = [] 
            frame_bboxes[frame_idx].append(np.array([float(z) for z in [y1,x1,y2,x2]]))
        
        for i, frame_idx in enumerate(frame_bboxes.keys()):
            frame_bboxes[frame_idx] = np.array(frame_bboxes[frame_idx]) 
            f = os.path.join(image_directory,'%s.png' % frame_idx) 
            im = cv.imread( f )
            _mode = 'train'
            if im is None:
                f = f.replace('/train/','/test/')
                im = cv.imread( f )
                _mode = 'test'
            
            parts = [ im[:,ii*w:(ii+1)*w,:] for ii in range(len_parts )]
            
            # scale to fit max_height
            #print('scaling',frame_bboxes[frame_idx][0],'to',(Hcomp/Hframe) * frame_bboxes[frame_idx][0])
            frame_bboxes[frame_idx] = (Hcomp/Hframe) * frame_bboxes[frame_idx]
            
            for j, (y1,x1,y2,x2) in enumerate(frame_bboxes[frame_idx]):
                # crop region around center of bounding box
                center = get_center(x1,y1,x2,y2,im.shape[0], im.shape[1], crop_dim)        

                try:
                    rois = [part[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:] for part in parts]
                    roi_comp = np.hstack(rois)
                    #if min(roi_comp.shape[:2])>1:
                    
                    f_roi = os.path.join(config['roi_dir'],_mode,'%s_%i.png' % (frame_idx, j))
                    #print(j,'coords',y1,x1,y2,x2,'center',center,roi_comp.shape)
                    #print(f_roi,roi_comp.shape)
                    if np.min(roi_comp.shape)>=3:
                        cv.imwrite(f_roi, roi_comp)
                            
                except Exception as e:
                    print(e)
        # </bboxes>
    
    #mean_rgb = calc_mean_rgb(config)
    hroi,wroicomp = cv.imread(glob(os.path.join(config['roi_dir'],'train','*.png'))[0]).shape[:2]
    wroi = hroi#wroicomp // (1+len(config['keypoint_names'])//3)
    print('wroi',hroi,wroicomp,wroi,'->',wroicomp//wroi)
    h = int(hroi * 0.98)
    config['img_height'] = 224
    config['img_width'] = 224

    from tensorflow.keras.applications.efficientnet import preprocess_input
    def load_im(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        comp = [ image[:,ii*wroi:(ii+1)*wroi,:] for ii in range(wroicomp//wroi)] # from hstacked to depth stacked
        
        # preprocess rgb image 
        comp[0] = unet.preprocess(comp[0])
        comp = tf.concat(comp,axis=2)
        comp = comp[:,:,:(3+len(config['keypoint_names']))]
        hh = h 
        if mode == 'train':
            # random scale augmentation
            if tf.random.uniform([]) > 0.3:
                hh = h + int(tf.random.uniform([],-h/10,h/10)) #h += int(tf.random.uniform([],-h/7,h/7))

            # apply augmentations
            # random rotation
            if tf.random.uniform([]) > 0.5:
                random_angle = tf.random.uniform([1],-35.*np.pi/180., 35.*np.pi/180.)  
                comp = tfa.image.rotate(comp, random_angle)

        # add background of heatmap
        background = 255 - tf.reduce_sum(comp[:,:,3:],axis=2)
        comp = tf.concat((comp,tf.expand_dims(background,axis=2)),axis=2)

        # crop
        #crop = tf.image.random_crop( comp, [hh,hh,1+3+len(config['keypoint_names'])])
        #crop = tf.image.resize(crop,[config['img_height'],config['img_width']])
        comp = tf.image.resize(comp,[int(tf.random.uniform([],0,10))+config['img_height'],int(tf.random.uniform([],0,10))+config['img_width']])
        crop = tf.image.random_crop( comp, [config['img_height'],config['img_width'],1+3+len(config['keypoint_names'])])
        # split stack into images and heatmaps
        image = crop[:,:,:3]
        
        #image = image - mean_rgb
        heatmaps = crop[:,:,3:] / 255.
        return image, heatmaps


    file_list = sorted(glob(os.path.join(config['roi_dir'],mode,'*.png')))
    file_list_tf = tf.data.Dataset.from_tensor_slices(file_list)
    data = file_list_tf.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['batch_size']).prefetch(4*config['batch_size'])#.cache()
    if mode == 'train':
        data = data.shuffle(512)
    return data 

def get_center(x1,y1,x2,y2,H,W,crop_dim):
    center = [int(round(y1+(y2-y1)/2.)),int(round(x1+(x2-x1)/2.))]
        
    center[0] = min(H-crop_dim//2,center[0] )
    center[0] = max(crop_dim//2,center[0] )
    center[1] = min(W-crop_dim//2,center[1] )
    center[1] = max(crop_dim//2,center[1] )
    return center 

def inference_heatmap(config, trained_model, frame, bounding_boxes):
    if 'max_height' in config and config['max_height'] is not None:
        H = config['max_height']
    else:
        H = frame.shape[0]
    W = frame.shape[1] * H/frame.shape[0]
    crop_dim = get_roi_crop_dim(config['project_id'], config['video_id'], H)

    y = None 
    for j, (y1,x1,y2,x2) in enumerate(bounding_boxes): 
        incoords = [y1,x1,y2,x2]
        # scale down bounding boxes 
        x1*=H/frame.shape[0]
        y1*=H/frame.shape[0]
        x2*=H/frame.shape[0]
        y2*=H/frame.shape[0]

        # crop region around center of bounding box
        center = get_center(x1,y1,x2,y2, H, W, crop_dim)
        
        roi = frame[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:]
        roi = tf.image.resize(roi,[config['img_height'],config['img_width']])
        roi = tf.expand_dims(tf.convert_to_tensor(roi),axis=0)
        yroi = trained_model.predict(roi)
        yroi = cv.resize(yroi,(crop_dim,crop_dim))

        # paste onto whole frame y 
        if y is None:
            y = np.zeros([frame.shape[0],frame.shape[1],y.shape[-1]])

        y[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:] += yroi 
    return y 
    
def train(config):
    config['lr'] = 1e-4
    config['cutmix'] = True
    print('[*] config', config)
    
    
    dataset_train = load_roi_dataset(config,mode='train')
    dataset_test = load_roi_dataset(config,mode='test')

    net = unet.get_model(config) # outputs: keypoints + background
    
    # decaying learning rate 
    decay_steps, decay_rate = 3000, 0.95
    lr = tf.keras.optimizers.schedules.ExponentialDecay(config['lr'], decay_steps, decay_rate)
    optimizer = tf.keras.optimizers.Adam(lr)

    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    checkpoint_path = os.path.expanduser("~/checkpoints/roi_keypoint/%s-%s" % (config['project_name'],now))

    writer_train = tf.summary.create_file_writer(checkpoint_path+'/train')
    writer_test = tf.summary.create_file_writer(checkpoint_path+'/test')
      
    ckpt = tf.train.Checkpoint(net = net, optimizer = optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

    def train_step(inp, y, writer_train, writer_test, global_step, should_summarize = False):
        with tf.GradientTape(persistent=True) as tape:
            predicted_heatmaps = net(inp,training=True)[0]
            loss = loss_func(y,predicted_heatmaps)
        
        # clipped gradients
        gradients = tape.gradient(loss,net.trainable_variables)
        gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        
        # update weights
        optimizer.apply_gradients(zip(gradients,net.trainable_variables))


        should_test = global_step % 250 == 0
        test_loss = 0.0
        if should_test:
            # test data
            nt = 0
            for xt,yt in dataset_test:
                predicted_test = net(xt,training=False)[0]
                if not predicted_test.shape[1] == y.shape[1]:
                    predicted_test = tf.image.resize(predicted_test, x.shape[1:3]) 

                test_loss += loss_func(predicted_test, yt)
                nt += 1 
            test_loss = test_loss / nt 

        # write summary
        if should_summarize:
            with tf.device("cpu:0"):
                def im_summary(name,data):
                    tf.summary.image(name,data,step=global_step)
                with writer_train.as_default():
                    tf.summary.scalar("loss %s" % config['loss'],loss,step=global_step)
                    tf.summary.scalar('min',tf.reduce_min(predicted_heatmaps[:,:,:,:-1]),step=global_step)
                    tf.summary.scalar('max',tf.reduce_max(predicted_heatmaps[:,:,:,:-1]),step=global_step)
                    im_summary('image',inp/256.)
                    for kk, keypoint_name in enumerate(config['keypoint_names']):
                        im_summary('heatmap_%s' % keypoint_name, tf.concat((tf.expand_dims(y[:,:,:,kk],axis=3), tf.expand_dims(predicted_heatmaps[:,:,:,kk],axis=3)),axis=2))
                    im_summary('background', tf.concat((tf.expand_dims(y[:,:,:,-1],axis=3),tf.expand_dims(predicted_heatmaps[:,:,:,-1],axis=3)),axis=2))
                    
                    tf.summary.scalar("learning rate", lr(global_step), step = global_step)
                    writer_train.flush()    

                if should_test:            
                    with writer_test.as_default():
                        tf.summary.scalar("loss %s" % config['loss'],test_loss,step=global_step)
                        im_summary('image',xt/256.)
                        
                        for kk, keypoint_name in enumerate(config['keypoint_names']):
                            im_summary('heatmap_%s' % keypoint_name, tf.concat((tf.expand_dims(yt[:,:,:,kk],axis=3), tf.expand_dims(predicted_test[:,:,:,kk],axis=3)),axis=2))
                        
                        im_summary('background', tf.concat((tf.expand_dims(yt[:,:,:,-1],axis=3),tf.expand_dims(predicted_test[:,:,:,-1],axis=3)),axis=2))
                            
                        writer_test.flush()

        
        result = {'train_loss':loss}
        if should_test:
            result['test_loss'] = test_loss 
        return result 

    n = 0
    swaps = model.get_swaps(config)
    test_losses = []
    
    ddata = {}
    import pickle 
    ttrainingstart = time.time()
    epoch = -1
    early_stopping = False
    while True:
        epoch += 1 
        start = time.time()
        epoch_steps = 0
        epoch_loss = 0.0

        if 1:# try:
            for x,y in dataset_train:
                if 1:
                    if np.random.random() < 0.5:
                        x,y = model.hflip(swaps,x,y)
                    if np.random.random() < 0.5:
                        x,y = model.vflip(swaps,x,y)
                if 1:        
                    # mixup augmentation
                    if np.random.random() < 0.9:
                        if config['mixup'] and np.random.random() > 0.5:
                            x, y = model.mixup(x,y) 
                        else:
                            if config['cutmix'] and np.random.random() > 0.5:
                                x, y = model.cutmix(x,y)
                    
                should_summarize=n%100==0
                step_result = train_step(x, y, writer_train, writer_test, n, should_summarize=should_summarize)
                
                if n % 2000 == 0:
                    ckpt_save_path = ckpt_manager.save()
                    print('[*] saving model to %s'%ckpt_save_path)
                    net.save(os.path.join(checkpoint_path,'trained_model.h5'))
                
                if 'test_loss' in step_result:
                    test_losses.append(step_result['test_loss'])
                
                finish = False
                if n == config['max_steps']-1:
                    print('[*] stopping keypoint estimation after step %i, because computational budget run out.' % n)
                    finish = True 
                
                if 'test_loss' in step_result and config['early_stopping'] and len(test_losses) > 3:
                    # early stopping        
                    if step_result['test_loss'] > test_losses[-2] and step_result['test_loss'] > test_losses[-3] and step_result['test_loss'] > test_losses[-4]:
                        finish = True 
                        print('[*] stopping keypoint estimation early at step %i, because current test loss %f is higher than previous %f and %f' % (n, test_losses[-1], test_losses[-2], test_losses[-3]))
                
                if finish:
                    return True 
            
                n+=1
        #except Exception as e:
        #    print('step',n,'\n',e)
                
def main(args):
    config = model.get_config(args.project_id)
    config['video_id'] = int(args.video_id)

    print(config,'\n')
    model.create_train_dataset(config)
    checkpoint_path = train(config)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    main(args)
