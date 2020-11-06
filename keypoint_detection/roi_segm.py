"""
    detect keypoints on cropped animals

"""

import tensorflow as tf
import tensorflow_addons as tfa
import os
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from datetime import datetime 
from IPython.display import clear_output
from multitracker.keypoint_detection import model , unet, heatmap_drawing, focus_augmentation, stacked_hourglass
from multitracker.be import video 
import cv2 as cv 
from glob import glob 
from random import shuffle
import numpy as np 
from datetime import datetime 
import json 

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

focal_loss = tfa.losses.SigmoidFocalCrossEntropy(False)
cce_loss = tf.keras.losses.CategoricalCrossentropy(False)
l2_loss = tf.keras.losses.MeanSquaredError()

def calc_focal_loss(ytrue,ypred):
    return tf.reduce_mean( focal_loss(ytrue, ypred))

def calc_cce_loss(ytrue, ypred):
    return tf.reduce_mean( cce_loss(ytrue, ypred) )

def calc_l2_loss(ytrue, ypred):
    return tf.reduce_mean( l2_loss(ytrue, ypred) )

def calc_accuracy(config, ytrue, ypred):
    correct_prediction = tf.equal(tf.argmax(ytrue,3),tf.argmax(ypred,3))
    correct_prediction = tf.cast(correct_prediction, "float")
    acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return acc 


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
    crop_dim = np.percentile(deltas, 97) * 1.1
    
    # scale to target frame height
    crop_dim = int(crop_dim * Htarget/Hframe) 

    # not bigger than image 
    crop_dim = min(crop_dim,Htarget)
    return crop_dim
    #return int(Htarget/3.)

def write_crop_to_disk(obj):
    f = obj['f']
    crop_dim_extended = obj['crop_dim_extended']
    config = obj['config']
    w, Hcomp,Hframe = obj['w'],obj['Hcomp'],obj['Hframe']
    im = cv.imread( f )
    _mode = 'train'
    if im is None:
        f = f.replace('/train/','/test/')
        im = cv.imread( f )
        _mode = 'test'
    parts = [ im[:,ii*w:(ii+1)*w,:] for ii in range(obj['len_parts'] )]
    #print(i,frame_idx,'boxes',frame_bboxes[frame_idx])
    # scale to fit max_height
    #print('scaling',frame_bboxes[frame_idx][0],'to',(Hcomp/Hframe) * frame_bboxes[frame_idx][0])
    obj['boxes'] = (Hcomp/Hframe) * obj['boxes']
    
    add_backgrounds = True
    #obj['boxes'] = [] # WARNING! only for bg debug
    if add_backgrounds: 
        ## add 3 random background patches without keypoints
        num_random_backgrounds = 3 
        while num_random_backgrounds > 0:
            #random_box = sorted(tuple(np.int32(np.around(np.random.uniform(0,im.shape[1]))))) + sorted(tuple(np.int32(np.around(np.random.uniform(0,im.shape[0])))))
            #random_box = [random_box[2],random_box[0],random_box[3],random_box[1]] # (x1,x2,y1,y2)=>(y1,x1,y2,x2)
            rx, ry = int(np.random.uniform(crop_dim_extended//2,im.shape[0]-crop_dim_extended//2)),int(np.random.uniform(crop_dim_extended//2,im.shape[1]-crop_dim_extended//2))
            random_box = [ry-crop_dim_extended//2,rx-crop_dim_extended//2,ry+crop_dim_extended//2,rx+crop_dim_extended//2]

            # if not overlaying with other boxes, add as background patch
            overlapping = True 
            for ii in range(len(obj['boxes'])):    # l1 = randombox, l2 = 
                positive_box = obj['boxes'][ii]
                if random_box[1]>=positive_box[3] or positive_box[1]>=random_box[3]:
                    overlapping = False 
                elif random_box[0] <= positive_box[2] or positive_box[2] <= random_box[0]:
                    overlapping = False

            if not overlapping:
                obj['boxes'] = np.vstack((obj['boxes'],random_box))
                num_random_backgrounds-=1

    for j, (y1,x1,y2,x2) in enumerate(obj['boxes']):
        # crop region around center of bounding box
        center = get_center(x1,y1,x2,y2,im.shape[0], im.shape[1], crop_dim_extended)        

        try:
            rois = [part[center[0]-crop_dim_extended//2:center[0]+crop_dim_extended//2,center[1]-crop_dim_extended//2:center[1]+crop_dim_extended//2,:] for part in parts]
            roi_comp = np.hstack(rois)
            #if min(roi_comp.shape[:2])>1:
            
            f_roi = os.path.join(config['roi_dir'],_mode,'%i_%s_%i.png' % (obj['video_id'], obj['frame_idx'], j))
            if np.min(roi_comp.shape[:2])>=3:# and roi_comp.shape[1]==(len_parts * crop_dim):
                cv.imwrite(f_roi, roi_comp)
                    
        except Exception as e:
            print(e)

def filter_crop_shape(obj):
    im = cv.imread(obj['f'])
    if not (im.shape[0] == obj['H'] and im.shape[1] == obj['W']):
        os.remove(obj['f'])
        return True 
    return False 

def load_roi_dataset(config,mode='train',batch_size=None):
    if mode=='train' or mode == 'test':
        image_directory = os.path.join(config['data_dir'],'%s' % mode)
    [Hframe,Wframe,_] = cv.imread(glob(os.path.join(os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'test'),'*.png'))[0]).shape
    
    [Hcomp,Wcomp,_] = cv.imread(glob(os.path.join(image_directory,'*.png'))[0]).shape
    H = Hcomp 
    w = int(Wframe*Hcomp/Hframe)

    len_parts = Wcomp // w  
    crop_dim = get_roi_crop_dim(config['project_id'], config['video_id'], Hcomp)
    crop_dim_extended_ratio = 1.5
    crop_dim_extended = min(Hcomp, crop_dim * crop_dim_extended_ratio)
    crop_dim_extended = int(crop_dim_extended)
    crop_dim_extended_ratio = crop_dim_extended / crop_dim 
    #print('crop_dim_extended_ratio',crop_dim_extended_ratio,'crop_dim',crop_dim,'crop_dim_extended',crop_dim_extended)
    
    if batch_size is None:
        batch_size = config['batch_size']

    for _mode in ['train','test']:
        if not os.path.isdir(os.path.join(config['roi_dir'],_mode)): os.makedirs(os.path.join(config['roi_dir'],_mode))
        
    if 0:
        for ii,ff in enumerate(glob(os.path.join(config['roi_dir'],'train','*.png'))):
            print(ii,ff.split('/')[-1],cv.imread(ff).shape)
    
    if len(glob(os.path.join(config['roi_dir'],'train','*.png')))==0:
        print('[*] creating cropped regions for each animal to train keypoint prediction ...')
        # extract bounding boxes of animals
        # <bboxes>
        for _video_id in config['train_video_ids'].split(','):
            _video_id = int(_video_id)
            frame_bboxes = {}
            db.execute("select * from bboxes where video_id=%i;" % _video_id)
            db_boxxes = [x for x in db.cur.fetchall()]
            shuffle(db_boxxes)
            for dbbox in db_boxxes:
                _, _, frame_idx, x1, y1, x2, y2 = dbbox 
                if not frame_idx in frame_bboxes:
                    frame_bboxes[frame_idx] = [] 
                frame_bboxes[frame_idx].append(np.array([float(z) for z in [y1,x1,y2,x2]]))
            
            with Pool(processes=os.cpu_count()) as pool:
                result_objs=[]
            
                for i, frame_idx in enumerate(frame_bboxes.keys()):
                    frame_bboxes[frame_idx] = np.array(frame_bboxes[frame_idx]) 
                    f = os.path.join(image_directory,'%i_%s.png' % (_video_id,frame_idx)) 
                    obj = {'Hframe':Hframe,'Hcomp':Hcomp,'w':w,'f':f, 'video_id':_video_id, 'config':config,'crop_dim_extended':crop_dim_extended,'len_parts':len_parts,'frame_idx':frame_idx,'boxes':frame_bboxes[frame_idx]}
                    result_objs.append(pool.apply_async(write_crop_to_disk,(obj,)))
                    #write_crop_to_disk(obj)
                    
                results = [result.get() for result in result_objs]
            
        with Pool(processes=os.cpu_count()) as pool:
            # check homogenous image sizes
            results_shapefilter = []
            sampled_shapes = []
            _roicrops = glob(os.path.join(config['roi_dir'],'train','*.png'))
            for ii in range(min(100,len(_roicrops))):
                sampled_shapes.append(cv.imread(_roicrops[int(np.random.uniform() * len(_roicrops))]).shape[:2])
            sampled_H, sampled_W = np.median(np.array(sampled_shapes)[:,0]), np.median(np.array(sampled_shapes)[:,1])
            for ii,ff in enumerate(glob(os.path.join(config['roi_dir'],mode,'*.png'))): 
                results_shapefilter.append( pool.apply_async(filter_crop_shape,({'f':ff,'H':sampled_H, 'W':sampled_W},)) )
            results_shapefilter = [result.get() for result in results_shapefilter]
        # </bboxes>weights='imagenet'
    

    
    #mean_rgb = calc_mean_rgb(config)
    hroi,wroicomp = cv.imread(glob(os.path.join(config['roi_dir'],'train','*.png'))[0]).shape[:2]
    wroi = hroi#wroicomp // (1+len(config['keypoint_names'])//3)
    h = hroi #int(hroi * 0.98)
    if 1:
        print('wroi',hroi,wroicomp,wroi,'->',wroicomp//wroi)
    def load_im(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        
        comp = [ image[:,ii*wroi:(ii+1)*wroi,:] for ii in range(wroicomp//wroi)] # from hstacked to depth stacked
        comp = [comp[0]] + [c[:,:,::-1] for c in comp[1:]] # keypoint heatmaps are in BGR not RGB

        # preprocess rgb image 
        comp[0] = unet.preprocess(config, comp[0])
        comp = tf.concat(comp,axis=2)
        comp = comp[:,:,:(3+len(config['keypoint_names']))] # cut 'overhanging' channels, that were filled up to reach 3channel png image
        hh = h 
        if mode == 'train' and config['rotation_augmentation']:
            # random scale augmentation
            #if tf.random.uniform([]) > 0.3:
            #    hh = h + int(tf.random.uniform([],-h/10,h/10)) #h += int(tf.random.uniform([],-h/7,h/7))

            # apply augmentations
            # random rotation
            if tf.random.uniform([]) > 0.5:
                random_angle = tf.random.uniform([1],-35.*np.pi/180., 35.*np.pi/180.)  
                comp = tfa.image.rotate(comp, random_angle)

        # add background of heatmap
        background = 255 - tf.reduce_sum(comp[:,:,3:],axis=2)
        comp = tf.concat((comp,tf.expand_dims(background,axis=2)),axis=2)

        # scale down a bit bigger than need before random cropping
        comp = tf.image.resize(comp,[int(config['img_height']*crop_dim_extended_ratio),int(config['img_width']*crop_dim_extended_ratio)])
        # random crop to counteract imperfect bounding box centers
        crop = tf.image.random_crop( comp, [config['img_height'],config['img_width'],1+3+len(config['keypoint_names'])])
    
        # split stack into images and heatmaps
        image = crop[:,:,:3]
        
        #image = image - mean_rgb
        heatmaps = crop[:,:,3:] / 255.
        return image, heatmaps

    if mode == 'train' and 'experiment' in config and config['experiment']=='A':
        file_list = glob(os.path.join(config['roi_dir'],mode,'*.png'))
        shuffle(file_list)
        oldlen = len(file_list)
        file_list = file_list[:int(len(file_list) * config['data_ratio'])]
        file_list = sorted(file_list)
        print('[*] cutting training data from %i samples to %i samples' % (oldlen, len(file_list)))
    else:
        file_list = sorted(glob(os.path.join(config['roi_dir'],mode,'*.png')))
    
    print('[*] loaded %i samples for mode %s' % (len(file_list),mode))

    file_list_tf = tf.data.Dataset.from_tensor_slices(file_list)
    data = file_list_tf.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(4*batch_size)#.cache()
    #if mode == 'train':
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
    #config['cutmix'] = False
    #config['mixup'] = True
    if 'hourglass' in config['backbone']:
        config['num_hourglass'] = int(config['backbone'][9:])
        config['backbone'] = 'efficientnetLarge'
    print('[*] config', config)
    
    
    dataset_train = load_roi_dataset(config,mode='train')
    dataset_test = load_roi_dataset(config,mode='test')

    if config['num_hourglass'] == 1:
        net = unet.get_model(config) # outputs: keypoints + background
    else:
        net = stacked_hourglass.get_model(config)
        
    # decaying learning rate 
    decay_steps, decay_rate = 3000, 0.95
    lr = tf.keras.optimizers.schedules.ExponentialDecay(config['lr'], decay_steps, decay_rate)
    optimizer = tf.keras.optimizers.Adam(lr)

    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    if not 'experiment' in config:
        checkpoint_path = os.path.expanduser("~/checkpoints/roi_keypoint/%s-%s" % (config['project_name'],now))
    elif config['experiment'] == 'A':
        checkpoint_path = os.path.expanduser("~/checkpoints/experiments/%s/A/%i-%s" % (config['project_name'], int(100. * config['data_ratio']) , now))
    elif config['experiment'] == 'B':
        checkpoint_path = os.path.expanduser("~/checkpoints/experiments/%s/B/%s-%s" % (config['project_name'], ['random','imagenet'][int(config['should_init_pretrained'])] , now))
    elif config['experiment'] == 'C':
        checkpoint_path = os.path.expanduser("~/checkpoints/experiments/%s/C/%s-%s" % (config['project_name'], config['backbone'] , now))
        if config['num_hourglass'] > 1:
            checkpoint_path = checkpoint_path.replace('/C/','/C/hourglass-%i-'%config['num_hourglass'])
    elif config['experiment'] == 'D':
        checkpoint_path = os.path.expanduser("~/checkpoints/experiments/%s/D/%s-%s" % (config['project_name'], config['train_loss'] , now))
        
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    # write config as JSON
    file_json = os.path.join(checkpoint_path,'config.json')
    with open(file_json, 'w') as f:
        json.dump(config, f, indent=4)
    
    writer_train = tf.summary.create_file_writer(checkpoint_path+'/train')
    writer_test = tf.summary.create_file_writer(checkpoint_path+'/test')
    csv_train = os.path.join(checkpoint_path,'train_log.csv')
    csv_test = os.path.join(checkpoint_path,'test_log.csv')

    ckpt = tf.train.Checkpoint(net = net, optimizer = optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

    def train_step(inp, y, writer_train, writer_test, global_step, should_summarize = False):
        with tf.GradientTape(persistent=True) as tape:
            all_predicted_heatmaps = net(inp,training=True)
            loss = 0.0
            for predicted_heatmaps in all_predicted_heatmaps:
                if config['train_loss'] == 'focal':
                    loss += calc_focal_loss(y,predicted_heatmaps)
                elif config['train_loss'] == 'cce':
                    loss += calc_cce_loss(y,predicted_heatmaps)
                elif config['train_loss'] == 'l2':
                    loss += calc_l2_loss(y,predicted_heatmaps)
            loss = loss / float(len(all_predicted_heatmaps))

        # clipped gradients
        gradients = tape.gradient(loss,net.trainable_variables)
        gradients = [tf.clip_by_value(grad, -1, 1) for grad in gradients]
        
        # update weights
        optimizer.apply_gradients(zip(gradients,net.trainable_variables))

        should_test = global_step % 250 == 0
        test_losses = {'focal':0.0,'cce':0.0}
        test_accuracy = 0.0
        if should_test:
            # test data
            nt = 0
            for xt,yt in dataset_test:
                predicted_test = net(xt,training=False)[-1]
                if not predicted_test.shape[1] == yt.shape[1]:
                    predicted_test = tf.image.resize(predicted_test, x.shape[1:3]) 

                if 'focal' in config['test_losses']:
                    test_losses['focal'] += calc_focal_loss(yt,predicted_test)
                if 'cce' in config['test_losses']:
                    test_losses['cce'] += calc_cce_loss(yt,predicted_test)
                test_accuracy += calc_accuracy(config, yt,predicted_test)
                nt += 1 
            test_losses['focal'] = test_losses['focal'] / nt
            test_losses['cce'] = test_losses['cce'] / nt
            test_accuracy = test_accuracy / nt 


        # write summary
        if should_summarize:
            with tf.device("cpu:0"):
                def im_summary(name,data):
                    tf.summary.image(name,data,step=global_step)
                with writer_train.as_default():
                    tf.summary.scalar("loss %s" % config['train_loss'],loss,step=global_step)
                    tf.summary.scalar('min',tf.reduce_min(predicted_heatmaps[:,:,:,:-1]),step=global_step)
                    tf.summary.scalar('max',tf.reduce_max(predicted_heatmaps[:,:,:,:-1]),step=global_step)
                    accuracy = calc_accuracy(config, y,predicted_heatmaps)
                    tf.summary.scalar('accuracy', accuracy,step=global_step)
                    im_summary('image',inp/256.)
                    for kk, keypoint_name in enumerate(config['keypoint_names']):
                        im_summary('heatmap_%s' % keypoint_name, tf.concat((tf.expand_dims(y[:,:,:,kk],axis=3), tf.expand_dims(predicted_heatmaps[:,:,:,kk],axis=3)),axis=2))
                        #tf.summary.scalar(keypoint_name+'_gt_min',tf.reduce_min(y[:,:,:,kk]),step=global_step)
                        #tf.summary.scalar(keypoint_name+'_gt_max',tf.reduce_max(y[:,:,:,kk]),step=global_step)
                        #tf.summary.scalar(keypoint_name+'_pr_min',tf.reduce_min(predicted_heatmaps[:,:,:,kk]),step=global_step)
                        #tf.summary.scalar(keypoint_name+'_pr_max',tf.reduce_max(predicted_heatmaps[:,:,:,kk]),step=global_step)
                    im_summary('background', tf.concat((tf.expand_dims(y[:,:,:,-1],axis=3),tf.expand_dims(predicted_heatmaps[:,:,:,-1],axis=3)),axis=2))
                    
                    tf.summary.scalar("learning rate", lr(global_step), step = global_step)
                    writer_train.flush()    

                with open(csv_train,'a+') as ftrain:
                    ftrain.write('%i,%f,%f\n' % (global_step, loss, accuracy))

                if should_test:            
                    with writer_test.as_default():
                        for k,v in test_losses.items():
                            tf.summary.scalar("loss %s" % k,v,step=global_step)
                        tf.summary.scalar('accuracy', test_accuracy,step=global_step)
                        im_summary('image',xt/256.)
                        
                        for kk, keypoint_name in enumerate(config['keypoint_names']):
                            im_summary('heatmap_%s' % keypoint_name, tf.concat((tf.expand_dims(yt[:,:,:,kk],axis=3), tf.expand_dims(predicted_test[:,:,:,kk],axis=3)),axis=2))
                        
                        im_summary('background', tf.concat((tf.expand_dims(yt[:,:,:,-1],axis=3),tf.expand_dims(predicted_test[:,:,:,-1],axis=3)),axis=2))
                            
                        writer_test.flush()

                    with open(csv_test,'a+') as ftest:
                        ftest.write('%i,%f,%f,%f\n' % (global_step, test_losses['focal'],test_losses['cce'],test_accuracy))
        
        result = {'train_loss':loss}
        if should_test:
            result['test_loss'] = tf.reduce_sum([v for v in test_losses.values()])
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

        if 1:#try:
            for x,y in dataset_train:
            
                if np.random.random() < 0.5 and config['hflips']:
                    x,y = model.hflip(swaps,x,y)
                if np.random.random() < 0.5 and config['vflips']:
                    x,y = model.vflip(swaps,x,y)
                if np.random.random() < 0.5 and 'rot90s' in config and config['rot90s']:
                    _num_rots = 1+int(np.random.uniform(3))
                    for ir in range(_num_rots):
                        x = tf.image.rot90(x)
                        y = tf.image.rot90(y)
                if 1:        
                    # mixup augmentation
                    if np.random.random() < 0.5:
                        if config['mixup'] and np.random.random() > 0.5:
                            x, y = model.mixup(x,y) 
                        else:
                            if config['cutmix'] and np.random.random() > 0.5:
                                x, y = model.cutmix(x,y)
                            if config['mixup'] and np.random.random() > 0.5:
                                x, y = model.mixup(x,y)
                    #x = focus_augmentation.out_of_focus_augment(x, proba = 0.5)

                should_summarize=n%200==0
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
                
                if n>config['min_steps_keypoints'] and 'test_loss' in step_result and config['early_stopping'] and len(test_losses) > 3:
                    # early stopping        
                    if step_result['test_loss'] > test_losses[-2] and step_result['test_loss'] > test_losses[-3] and step_result['test_loss'] > test_losses[-4] and min(test_losses[:-1]) < 1.5*test_losses[-1]:
                        finish = True 
                        print('[*] stopping keypoint estimation early at step %i, because current test loss %f is higher than previous %f and %f' % (n, test_losses[-1], test_losses[-2], test_losses[-3]))
                
                if finish:
                    ckpt_save_path = ckpt_manager.save()
                    print('[*] saving model to %s'%ckpt_save_path)
                    net.save(os.path.join(checkpoint_path,'trained_model.h5'))
                    #return os.path.join(checkpoint_path,'trained_model.h5') 
                    return checkpoint_path 

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
