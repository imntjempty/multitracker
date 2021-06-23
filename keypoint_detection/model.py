
"""

    python3.7 -m multitracker.keypoint_detection.model --project_id 7 --video_id 9
"""

import os
import numpy as np 
try:
    import tensorflow as tf 
    import tensorflow_addons as tfa
except:
    print('[* model.py] not loading tensorflow')
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 

from multitracker.be import dbconnection

db = dbconnection.DatabaseConnection()
from multitracker.be import video 
    

def get_loss(predicted_heatmaps, y, config, mode = "train"):
    if mode == "test":
        return tf.reduce_mean(tfa.losses.SigmoidFocalCrossEntropy(False)(y, predicted_heatmaps))

    if config['loss'] == 'l1':
        loss = tf.reduce_mean(tf.abs(predicted_heatmaps - y))

    elif config['loss'] == 'l2':
        #loss = tf.reduce_mean(tf.nn.l2_loss(predicted_heatmaps - y)) / 5000.
        mse = tf.keras.losses.MeanSquaredError()
        loss = tf.reduce_mean( mse(y, predicted_heatmaps) )
    
    elif config['loss'] == 'dice':
        a = 2 * tf.reduce_sum(predicted_heatmaps * y, axis=-1 )
        b = tf.reduce_sum(predicted_heatmaps + y, axis=-1 )
        loss = tf.reduce_mean(1 - (a+1)/(b+1))

    elif config['loss'] == 'focal':
        loss_func = tfa.losses.SigmoidFocalCrossEntropy(False)
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

def get_model(config, verbose = False):
    from multitracker.keypoint_detection import stacked_hourglass, unet
    if config['kp_num_hourglass'] == 1:
        encoder, net = unet.get_model(config)
    else:
        encoder, net = stacked_hourglass.get_model(config)
    if verbose:
        #encoder.summary()
        net.summary()
        #for i, l in enumerate(net.layers):
        #    print('layer %i:'%i,l.name,l.output.shape,l.trainable)
    return net
# </network architecture>

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

def swap_leftright_channels(swaps, y):
    # swap corresponding left/right channels
    for i,j in swaps:
        channels = tf.unstack (y, axis=-1)
        restacked = [] 
        for k in range(len(channels)):
            if k==i:
                restacked.append(channels[j])
            elif k==j:
                restacked.append(channels[i])
            else:
                restacked.append(channels[k])
        y = tf.stack(restacked, axis=-1)
    return y 

def hflip(swaps, x, y):
    hx = x[:,:,::-1,:]
    hy = y[:,:,::-1,:]
    hy = swap_leftright_channels(swaps, hy)
    return hx, hy

def vflip(swaps, x, y):
    vx = x[:,::-1,:,:]
    vy = y[:,::-1,:,:]
    vy = swap_leftright_channels(swaps, vy)
    return vx, vy

def get_swaps(config):
    swaps = []
    for i, keypoint_classnameA in enumerate(config['keypoint_names']):
        for j, keypoint_classnameB in enumerate(config['keypoint_names']):
            if 'left' in keypoint_classnameA and keypoint_classnameA.replace('left','right') == keypoint_classnameB:
                swaps.append((i,j))
    return swaps 

def update_config_object_detection(config):
    config['object_detection_backbonepath'] = {
        'efficient': 'efficientdet_d1_coco17_tpu-32',                        #  54ms 38.4mAP
        'ssd': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',                   #  46ms 34.3mAP
        'fasterrcnn': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8' # 206ms 37.7mAP
    }[config['object_detection_backbone']]
    config['object_detection_batch_size'] = {'efficient': 4, 'ssd': 4, 'fasterrcnn': 4}[config['object_detection_backbone']]
    config['inference_objectdetection_batchsize'] = {'efficient': 4, 'ssd': 4, 'fasterrcnn': 4}[config['object_detection_backbone']]
    config['lr_objectdetection'] = 0.0005 
    if config['object_detection_backbone'] == 'efficient':
        config['lr_objectdetection'] *= 10.
    config['object_pretrained'] = True
    config['maxsteps_objectdetection'] = 250000 #50000
    config['minsteps_objectdetection'] = 20000 #25000
    config['object_finetune_warmup'] = 1000
    config['object_augm_flip'] = bool(1)
    config['object_augm_rot90'] = bool(1)
    config['object_augm_gaussian'] = bool(0) # 1
    config['object_augm_image'] = bool(0)
    config['object_augm_mixup'] = bool(0) # 1
    config['object_augm_crop'] = bool(0)
    config['object_augm_stitch'] = bool(0)

    config['object_detection_resolution'] = [640,640]
    return config 

# </train>
def get_config(project_id = 3):
    config = {'batch_size': 64}
    config.update({'img_height': 224,'img_width': 224})
    config['keypoint_resolution'] = [ config[k] for k in ['img_height','img_width'] ]
    config['kp_max_steps'] = 200000
    config['kp_min_steps'] = 50000
    config['kp_lr'] = 2e-5 * 5   *5 *2.
    config['kp_lr'] = 2e-5 * 5
    config['kp_blurpool'] = True
    config['ae_pretrained_encoder'] = [False,True][1]

    config['kp_mixup'] = [False, True][0]
    config['kp_cutmix'] = [False, True][1]
    config['kp_hflips'] = [False,True][1]
    config['kp_vflips'] = [False,True][1]
    config['kp_rotation_augmentation'] = bool(1)
    config['kp_rot90s'] = bool(1)
    config['kp_im_noise'] = bool(1)
    config['kp_im_transf'] = bool(1)
    config['kp_num_hourglass'] = 2 #8
    config['kp_finetune_warmup'] = 5000
    config['fov'] = 0.75 # default 0.5
    
    config['n_blocks'] = 5
    config['early_stopping'] = True 

    config['max_height'] = 1024
    config['project_id'] = project_id
    config['project_name'] = db.get_project_name(project_id)
    config['data_dir'] = os.path.expanduser('~/data/multitracker')
    config['kp_data_dir'] = os.path.join(dbconnection.base_data_dir, 'projects/%i/data' % config['project_id'])
    config['kp_roi_dir'] = os.path.join(dbconnection.base_data_dir, 'projects/%i/data_roi' % config['project_id'])

    config['keypoint_names'] = db.get_keypoint_names(config['project_id'])

    config['kp_backbone'] = ["vgg16","efficientnet","efficientnetLarge",'psp'][3]
    config['inference_keypoint_batchsize'] = 16

    config['object_detection_backbone'] = ['efficient','ssd','fasterrcnn'][2] ## https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    config = update_config_object_detection(config)
    
    config['kp_train_loss'] = ['cce','focal'][1]
    config['kp_test_losses'] = ['focal'] #['cce','focal']

    return config 
