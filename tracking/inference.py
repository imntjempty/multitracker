
import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py


from multitracker import util 
from multitracker.keypoint_detection import heatmap_drawing, model 
from multitracker.keypoint_detection import predict
from multitracker.keypoint_detection.blurpool import BlurPool2D

def load_model(path_model):
    t0 = time.time()
    trained_model = tf.keras.models.load_model(h5py.File(os.path.join(path_model,'trained_model.h5'), 'r'),custom_objects={'BlurPool2D':BlurPool2D})
    t1 = time.time()
    print('[*] loaded keypoint model from %s in %f seconds.' %(path_model,t1-t0))
    return trained_model 

def load_data(project_id,video_id,max_minutes=0):
    frames_dir = predict.get_project_frame_train_dir(project_id, video_id)
    frame_files = sorted(glob(os.path.join(frames_dir,'*.png')))
    
    #frame_files = frame_files[int(np.random.uniform(2000)):]
    if max_minutes >0:
        nn = int(60*max_minutes*30 )
        ns = int(np.random.random()*(len(frame_files)-nn))
        #frame_files = frame_files[ ns:ns+nn ]
        frame_files = frame_files[:nn]

    if len(frame_files) == 0:
        raise Exception("ERROR: no frames found in " + str(frames_dir))
    print('[*] found %i frames' % len(frame_files))
    return frame_files

def get_heatmaps_keypoints(heatmaps, thresh_detection=0.5):
    x = np.array(heatmaps,copy=True)
    keypoints = [] 
    for c in range(x.shape[2]-1): # dont extract from background channel
        channel_candidates = predict.extract_frame_candidates(x[:,:,c], thresh = thresh_detection, pp = int(0.02 * np.min(x.shape[:2])))
        for [px,py,val] in channel_candidates:
            keypoints.append([px,py,c])

    # [debug] filter for nose 
    #keypoin1s = [kp for kp in keypoints if kp[2]==1]
    return keypoints 

def inference_heatmap(config, trained_model, frame):
    # predict whole image, height like trained height and variable width 
    # to keep aspect ratio and relative size        
    #bs, config['n_inferences'] = 1, 1
    bs, config['n_inferences'] = 4, 1
    w = 1+int((1./config['fov'])*config['img_height']/(float(frame.shape[0]) / frame.shape[1]))
    h = int((1./config['fov'])*config['img_height'])
    ggT = 2*2**config['n_blocks']
    if (w//ggT)*ggT-w>ggT//2:
        w = (w//ggT+1)*ggT 
    else:
        w = (w//ggT)*ggT 
    if (h//ggT)*ggT-h>ggT//2:
        h = (h//ggT+1)*ggT 
    else:
        h = (h//ggT)*ggT 

    xsmall = cv.resize(frame, (w,h))
    xsmall = tf.expand_dims(tf.convert_to_tensor(xsmall),axis=0)
    
    if bs > 1: 
        xsmall = tf.tile(xsmall,[bs,1,1,1])

    # 1) inference: run trained_model to get heatmap predictions
    tsb = time.time()
    if config['n_inferences'] == 1 and bs == 1:
        y = trained_model.predict(xsmall)
    else:
        y = trained_model(xsmall, training=True) / config['n_inferences']
        for ii in range(config['n_inferences']-1):
            y += trained_model(xsmall, training=True) / config['n_inferences']
    y = tf.reduce_mean(y,axis=[0]) # complete batch is of same image, so second dim average
    
    #if config['n_inferences'] > 1:
    try:
        y = cv.resize(y,tuple(frame.shape[:2][::-1]))
    except:
        y = y.numpy()
        y = cv.resize(y,tuple(frame.shape[:2][::-1]))
    #print('y',y.shape,frame.shape)
    
    tse = time.time() 
    return y


def get_keypoints_vis(frame, keypoints, keypoint_names):
    vis_keypoints = np.zeros(frame.shape,'uint8')
    
    # draw circles
    for [x,y,class_id,indv ] in keypoints:
        radius = np.min(vis_keypoints.shape[:2]) // 200
        px,py = np.int32(np.around([x,y]))
        # color by indv
        color = colors[int(indv)%len(colors)]
        c1,c2,c3 = color
        vis_keypoints = cv.circle(vis_keypoints,(px,py),radius,(int(c1),int(c2),int(c3)),-1)
    
    # draw labels
    for [x,y,class_id,indv ] in keypoints:
        px,py = np.int32(np.around([x,y]))
        color = colors[int(indv%len(colors))]
        name = "%i %s"%(int(indv),keypoint_names[int(class_id)])
        #cv.putText( vis_keypoints, name, (px+3,py-8), cv.FONT_HERSHEY_COMPLEX, 1, color, 3 )
    
    vis_keypoints = np.uint8(vis_keypoints//2 + frame//2)
    return vis_keypoints 