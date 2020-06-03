"""
    recognition video prediction
    load a trained model and predict and visualize each frame

    example call
        python3.7 -m multitracker.keypoint_detection.predict -model /home/alex/checkpoints/keypoint_tracking/2020-06-01_16-19-06 -project 2
"""

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py

from multitracker.keypoint_detection import heatmap_drawing, model 

# <network architecture>
from tensorflow.keras.applications.resnet_v2 import preprocess_input

def get_random_colors(n):
    colors = []
    for i in range(n):
        r = int(np.random.uniform(0,255))
        g = int(np.random.uniform(0,255))
        b = int(np.random.uniform(0,255))
        colors.append((r,g,b))
    colors = np.int32(colors)
    return colors 
colors = get_random_colors(100)

def get_project_frames(config, project_id = None):
    if project_id is None:
        project_id = config['project_id']
    
    def load_im(image_file):
        print(image_file)
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        return image 
    
    frames_dir = get_project_frame_test_dir(project_id)
    frames = tf.data.Dataset.list_files(os.path.join(frames_dir,'*.png'),shuffle=False).map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['batch_size'])#.prefetch(4*config['batch_size'])#.cache()
    return frames 

def get_project_frame_test_dir(project_id):
    return os.path.expanduser('~/data/multitracker/projects/%i/%i/frames/test' % (project_id,project_id))

def extract_frame_candidates(feature_map):
    step = -1
    max_step = 50
    stop_threshold_hit = False 
    frame_candidates = []
    while not stop_threshold_hit and step < max_step:
        step += 1
        # find new max pos
        max_pos = np.argmax(feature_map)
        py = max_pos // feature_map.shape[0]
        px = max_pos % feature_map.shape[1]
        val = np.max(feature_map)
        frame_candidates.append([px,py,val])

        # delete area around new max pos 
        pp = 5
        feature_map[py-pp:py+pp,px-pp:px+pp] = 0
        
        # stop extraction if max value has small probability 
        if val < 0.5:
            frame_candidates = frame_candidates[:-1]
            stop_threshold_hit = True 
    return frame_candidates

def predict(config, checkpoint_path, project_id):
    project_id = int(project_id)
    output_dir = '/tmp/multitracker/predictions/%i/%s' % (project_id, checkpoint_path.split('/')[-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
 
    t0 = time.time()
    path_model = os.path.join(checkpoint_path,'trained_model.h5')
    trained_model = tf.keras.models.load_model(h5py.File(path_model, 'r'))
    t1 = time.time()
    print('[*] loaded model from %s in %f seconds.' %(path_model,t1-t0))
    frames_dir = get_project_frame_test_dir(project_id)
    frame_files = sorted(glob(os.path.join(frames_dir,'*.png')))
    config['input_image_shape'] = cv.imread(frame_files[0]).shape[:2]

    config['n_inferences'] = 5
    
    frames = get_project_frames(config, project_id)
    print('[*] will predict %i frames (%f seconds of video)'%(len(frame_files),len(frame_files)/30.))

    cnt_output = 0 
    t2 = time.time()
    for ibatch, x in enumerate(frames):
        # first center crop with complete height of image and then rescale to target size 
        #xsmall = x[:,:, x.shape[2]//2 - x.shape[1]//2 : x.shape[2]//2 + x.shape[1]//2, : ]
        #xsmall = tf.image.resize(xsmall,(config['img_height'],config['img_width']))
        r = float(x.shape[1]) / x.shape[2]
        xsmall = tf.image.resize(x, (config['img_height'],1+int(config['img_height']/r)))

        # run trained_model to get heatmap predictions
        tsb = time.time()
        if config['n_inferences'] == 1:
            y = trained_model.predict(xsmall)
        else:
            y = trained_model(xsmall, training=True) / config['n_inferences']
            for ii in range(config['n_inferences']-1):
                y += trained_model(xsmall, training=True) / config['n_inferences']
        tse = time.time() 

        for b in range(x.shape[0]): # iterate through batch of frames
            # extract observation maxima positions for each channel of network prediction
            frame_candidates = [ [] for _ in range(y.shape[3]-1) ]
            for c in range(y.shape[3]-1):
                frame_idx = ibatch * x.shape[0] + b 
                feature_map = y[b,:,:,c].numpy()
                
                frame_candidates[c] = extract_frame_candidates(feature_map)
                
                if len(frame_candidates[c]) > 0:      
                    print('frame',frame_idx, config['keypoint_names'][c], ':', frame_candidates[c])

        should_write = 1 
        if should_write:
            for b in range(x.shape[0]):
                vis_frame = np.zeros([y.shape[1],y.shape[2],3])
                #vis_frame = np.zeros([config['img_height'],config['img_width'],3])
                for c in range(y.shape[3]-1): # iterate through each channel for this frame except background channel
                    feature_map = y[b,:,:,c]
                    feature_map = np.expand_dims(feature_map, axis=2)
                    feature_map = colors[c] * feature_map 
                    vis_frame += feature_map 
                vis_frame = np.around(vis_frame)
                vis_frame = np.uint8(vis_frame)
                #print('1vis_frame',vis_frame.shape,vis_frame.dtype,vis_frame.min(),vis_frame.max())
                overlay = np.uint8(xsmall[b,:,:,:]//2 + vis_frame//2)
                vis_frame = np.hstack((overlay,vis_frame))
                fp = os.path.join(output_dir,'predict-{:05d}.png'.format(cnt_output))
                cv.imwrite(fp, vis_frame)
                
                cnt_output += 1 

        # estimate duration until done with all frames
        if cnt_output % ( 30*16) == 0:
            dur_one = float(tse-t2) / cnt_output
            dur_left_minute = float(len(frame_files)-cnt_output) * dur_one / 60.
            print('[*] %i minutes left for predicting (%i/100 done)' % (dur_left_minute, int(cnt_output / len(frame_files) * 100)))



def main(checkpoint_path, project_id):
    config = model.get_config()
    predict(config, checkpoint_path, project_id)


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-model')
    parser.add_argument('-project')
    args = parser.parse_args()
    main(args.model, args.project)