
import os
from glob import glob 
import subprocess

import cv2 as cv 
import numpy as np
import tensorflow as tf 
assert tf.__version__.startswith('2.'), 'YOU MUST INSTALL TENSORFLOW 2.X'
print('[*] TF version',tf.__version__)
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import Model

from multitracker.tracking.deep_sort.application_util import preprocessing
from multitracker.tracking.deep_sort.application_util import visualization
from multitracker.tracking.deep_sort.deep_sort import nn_matching
from multitracker.tracking.deep_sort import deep_sort_app
from multitracker.tracking.deep_sort.deep_sort.detection import Detection
from multitracker.tracking.deep_sort.deep_sort.tracker import Tracker

from multitracker import autoencoder
from multitracker.keypoint_detection import roi_segm, unet
from multitracker.tracking.inference import get_heatmaps_keypoints
from multitracker.tracking.keypoint_tracking import tracker as keypoint_tracking
from multitracker.be import video
from multitracker import util 

from multitracker.tracking import inference

colors = util.get_colors()



"""
    implementation of Tracktor++ 
    Tracking without bells and whistles
    https://arxiv.org/pdf/1903.05625v3.pdf

    ported from official pytorch implementation https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py
"""

import os 
import numpy as np 


class Tracker(object):
    def __init__(self,config):
        self.config = config 
        self.cl = 1 # only one class
        self.tracks = []
        self.inactive_tracks = []

        self.track_num = 0
        self.results = {}
        self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def predict(self):
        pass # legacy function from DeepSORT
    
    def update(self, detections):
        pass
        # apply motion model

class Track(object):
    
    def __init__(self):
        self.last_detections = [] 
        self.last_means = [] 
    
    def update(self,detection):
        self.last_detections.append(detection)
        self.last_means.append(self.mean)



def run(config, detection_model, encoder_model, keypoint_model, crop_dim):
    min_confidence = 0.7
    min_confidence_keypoints = 0.6
    nms_max_overlap = 1.

    if 'video' in config and config['video'] is not None:
        video_reader = cv.VideoCapture( config['video'] )
    else:
        video_reader = None 
    
    [Hframe,Wframe,_] = cv.imread(glob(os.path.join(os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'test'),'*.png'))[0]).shape
    
    video_file_out = inference.get_video_output_filepath(config)
    if os.path.isfile(video_file_out): os.remove(video_file_out)
    import skvideo.io
    video_writer = skvideo.io.FFmpegWriter(video_file_out, outputdict={
        '-vcodec': 'libx264',  #use the h.264 codec
        '-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }) 

    seq_info = deep_sort_app.gather_sequence_info(config)
    visualizer = visualization.Visualization(seq_info, update_ms=5)
    print('[*] writing video file %s' % video_file_out)
    
    ## initialize tracker for boxes and keypoints
    tracker = Tracker(config)
    keypoint_tracker = keypoint_tracking.KeypointTracker()
    frame_directory = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'train')
    frame_idx = 0
    config['count'] = 0
    while video_reader.isOpened():
        ret, frame = video_reader.read()
    
        frame_idx += 1 
        config['count'] = frame_idx

        frame_, detections = inference.detect_frame_boundingboxes(config, detection_model, encoder_model, seq_info, frame, frame_idx)
        detections = [d for d in detections if d.confidence >= min_confidence]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        #print('[*] found %i detections' % len(detections))
        # Update tracker.
        tracker.predict()
        tracker.update(detections)
        
        keypoints = inference.inference_keypoints(config, frame, detections, keypoint_model, crop_dim, min_confidence_keypoints)
        # update tracked keypoints with new detections
        tracked_keypoints = keypoint_tracker.update(keypoints)

        print('%i - %i detections. %i keypoints' % (config['count'],len(detections), len(keypoints)),[kp for kp in keypoints])
        out = deep_sort_app.visualize(visualizer, frame, tracker, detections, keypoint_tracker, keypoints, tracked_keypoints, crop_dim)
        video_writer.writeFrame(cv.cvtColor(out, cv.COLOR_BGR2RGB)) #out[:,:,::-1])
        config['count'] += 1

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

