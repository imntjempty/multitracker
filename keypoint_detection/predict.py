"""
    recognition video prediction
    load a trained model and predict and visualize each frame

    example call
        python3.7 -m multitracker.keypoint_detection.predict --model /home/alex/checkpoints/keypoint_tracking/2020-06-01_16-19-06 --project_id 4 --video_id 5
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

from multitracker import util 
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

def get_project_frames(config, project_id = None, video_id = None):
    if project_id is None:
        project_id = config['project_id']
    if video_id is None:
        video_id = config['video_id']
    
    def load_im(image_file):
        print(image_file)
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        return image 
    
    frames_dir = get_project_frame_test_dir(project_id, video_id)
    frames = tf.data.Dataset.list_files(os.path.join(frames_dir,'*.png'),shuffle=False).map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['batch_size'])#.prefetch(4*config['batch_size'])#.cache()
    return frames 

def get_project_frame_test_dir(project_id, video_id):
    return os.path.expanduser('~/data/multitracker/projects/%i/%i/frames/test' % (project_id,video_id))

def extract_frame_candidates(feature_map, thresh = 0.3):
    step = -1
    max_step = 50
    stop_threshold_hit = False 
    frame_candidates = []
    while not stop_threshold_hit and step < max_step:
        step += 1
        # find new max pos
        max_pos = np.unravel_index(np.argmax(feature_map),feature_map.shape)
        py = max_pos[0] #max_pos // feature_map.shape[0]
        px = max_pos[1] #max_pos % feature_map.shape[1]
        val = np.max(feature_map)
        frame_candidates.append([px,py,val])

        # delete area around new max pos 
        pp = 5
        feature_map[py-pp:py+pp,px-pp:px+pp] = 0
        feature_map[py][px] = 0 
        
        # stop extraction if max value has small probability 
        if val < thresh:
            frame_candidates = frame_candidates[:-1]
            stop_threshold_hit = True 
    return frame_candidates

def point_distance(config, p1, p2):
    dist = np.linalg.norm(np.array(p1)/config['img_height']-np.array(p2)/config['img_height'])
    score = 1. / dist 
    score = min(1, score) # max 1
    return score 


def predict(config, checkpoint_path, project_id, video_id):
    project_id = int(project_id)
    output_dir = '/tmp/multitracker/predictions/%i/%s' % (project_id, checkpoint_path.split('/')[-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
 
    t0 = time.time()
    path_model = os.path.join(checkpoint_path,'trained_model.h5')
    trained_model = tf.keras.models.load_model(h5py.File(path_model, 'r'))
    t1 = time.time()
    print('[*] loaded model from %s in %f seconds.' %(path_model,t1-t0))
    frames_dir = get_project_frame_test_dir(project_id, video_id)
    frame_files = sorted(glob(os.path.join(frames_dir,'*.png')))
    if len(frame_files) == 0:
        raise Exception("ERROR: no frames found in " + str(frames_dir))
    config['input_image_shape'] = cv.imread(frame_files[0]).shape[:2]

    config['n_inferences'] = 5
    
    frames = get_project_frames(config, project_id, video_id)
    print('[*] will predict %i frames (%f seconds of video)'%(len(frame_files),len(frame_files)/30.))

    cnt_output = 0 
    t2 = time.time()
    
    history_candidates = []

    for ibatch, x in enumerate(frames):
        # first center crop with complete height of image and then rescale to target size 
        #xsmall = x[:,:, x.shape[2]//2 - x.shape[1]//2 : x.shape[2]//2 + x.shape[1]//2, : ]
        #xsmall = tf.image.resize(xsmall,(config['img_height'],config['img_width']))

        # predict whole image, height like trained height and variable width 
        # to keep aspect ratio and relative size        
        w = 1+int(2*config['img_height']/(float(x.shape[1]) / x.shape[2]))
        xsmall = tf.image.resize(x, (2*config['img_height'],w))

        # 1) inference: run trained_model to get heatmap predictions
        tsb = time.time()
        if config['n_inferences'] == 1:
            y = trained_model.predict(xsmall)
        else:
            y = trained_model(xsmall, training=True) / config['n_inferences']
            for ii in range(config['n_inferences']-1):
                y += trained_model(xsmall, training=True) / config['n_inferences']
        tse = time.time() 

        # 2) extract frame candidates
        for b in range(x.shape[0]): # iterate through batch of frames
            # extract observation maxima positions for each channel of network prediction
            frame_candidates = [ [] for _ in range(y.shape[3]-1) ]
            for c in range(y.shape[3]-1):
                frame_idx = ibatch * x.shape[0] + b 
                feature_map = y[b,:,:,c].numpy()
                
                frame_candidates[c] = extract_frame_candidates(feature_map)
                
                if len(frame_candidates[c]) > 0:      
                    print('frame',frame_idx, config['keypoint_names'][c], ':', frame_candidates[c])
            history_candidates.append( frame_candidates )

        # draw visualization and write to disk
        should_write = 1 
        if should_write:
            for b in range(x.shape[0]):
                vis_frame = np.zeros([y.shape[1],y.shape[2],3])
                for c in range(y.shape[3]-1): # iterate through each channel for this frame except background channel
                    feature_map = y[b,:,:,c]
                    feature_map = np.expand_dims(feature_map, axis=2)
                    # draw heatmap
                    feature_map = colors[c] * feature_map 
                    feature_map = np.uint8(np.around(feature_map))

                    # draw candidates circles
                    for cand in history_candidates[cnt_output][c]:
                        feature_map[cand[1]][cand[0]] = colors[c]
                        c1,c2,c3 = colors[c]
                        feature_map = cv.circle(feature_map,(cand[0],cand[1]),3,(int(c1),int(c2),int(c3)),-1)
                    vis_frame += feature_map 
                vis_frame = np.around(vis_frame)
                vis_frame = np.uint8(vis_frame)
                vis_frame = cv.resize(vis_frame,tuple(xsmall.shape[1:-1][::-1]))
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

    if should_write:
        # create video afterwards
        video_file = os.path.join(output_dir,'video.mp4')
        util.make_video(output_dir,video_file)

def main(checkpoint_path, project_id, video_id):
    project_id = int(project_id)
    video_id = int(video_id)

    config = model.get_config(project_id=project_id)
    config['batch_size'] = 16
    predict(config, checkpoint_path, project_id, int(video_id))


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--project_id')
    parser.add_argument('--video_id')
    args = parser.parse_args()
    main(args.model, args.project_id, args.video_id)