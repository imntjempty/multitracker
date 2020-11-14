
"""
    main program to track animals and their corresponding limbs on a video file

    python3.7 -m multitracker.tracking --project_id 7 --video_id 13 --objectdetection_model ~/checkpoints/bbox_detection/MiceTop_vid9/2020-10-28_01-15-29 --autoencoder_model ~/checkpoints/multitracker_ae_bbox/MiceTop/2020-10-28_08-04-51 --keypoint_model /home/alex/checkpoints/roi_keypoint/MiceTop-2020-11-03_01-02-15
"""

import os
import numpy as np 
import tensorflow as tf 
tf.get_logger().setLevel('INFO')
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py
import json 

from multitracker import util 
from multitracker.be import video
from multitracker.keypoint_detection import heatmap_drawing, model , roi_segm
from multitracker.keypoint_detection import predict 
from multitracker.tracking.inference import load_model as load_keypoint_model
from multitracker.tracking.inference import load_data, load_model, get_heatmaps_keypoints
from multitracker.keypoint_detection.roi_segm import get_center
from multitracker.tracking.tracklets import get_tracklets
from multitracker.tracking.clustering import get_clustlets
from multitracker.object_detection import finetune
from multitracker.tracking.deep_sort import deep_sort_app
from multitracker import autoencoder
from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

def main(args):
    tstart = time.time()
    config = model.get_config(project_id = args.project_id)
    config['project_id'] = args.project_id
    config['video_id'] = args.video_id
    config['keypoint_model'] = args.keypoint_model
    config['autoencoder_model'] = args.autoencoder_model 
    config['objectdetection_model'] = args.objectdetection_model
    config['train_video_ids'] = args.train_video_ids
    config['minutes'] = args.minutes
    config['fixed_number'] = db.get_video_fixednumber(args.video_id) #args.fixed_number
    config['fixed_number'] = None
    config['n_blocks'] = 4

    # <load frames>
    output_dir = '/tmp/multitracker/object_detection/predictions/%i' % (config['video_id'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('[*] writing object detection bounding boxes %f minutes of video %i frames to %s' % (config['minutes'],config['video_id'],output_dir))

    frames_dir = os.path.join(dbconnection.base_data_dir, 'projects/%i/%i/frames/train' % (config['project_id'], config['video_id']))
    frame_files = sorted(glob(os.path.join(frames_dir,'*.png')))
    if len(frame_files) == 0:
        raise Exception("ERROR: no frames found in " + str(frames_dir))
    
    if config['minutes']> 0:
        frame_files = frame_files[:int(30. * 60. * config['minutes'])]

    # </load frames>

    # <train models>
    # 1) animal bounding box finetuning -> trains and inferences 
    config['objectdetection_max_steps'] = 30000
    # train object detector
    now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
    checkpoint_directory_object_detection = os.path.expanduser('~/checkpoints/multitracker/bbox/vids%s-%s' % (config['train_video_ids'], now))
    object_detect_restore = None 
    if 'objectdetection_model' in config and config['objectdetection_model'] is not None:
        object_detect_restore = config['objectdetection_model']
    
    detection_model = None
    if object_detect_restore is None:
        detection_model = finetune.finetune(config, checkpoint_directory_object_detection, checkpoint_restore = object_detect_restore)
        print('[*] trained object detection model',checkpoint_directory_object_detection)
        config['object_detection_model'] = checkpoint_directory_object_detection

    ## crop bbox detections and train keypoint estimation on extracted regions
    #point_classification.calculate_keypoints(config, detection_file_bboxes)
    
    # 2) train autoencoder for tracking appearence vector
    if config['autoencoder_model'] is None:
        config_autoencoder = autoencoder.get_autoencoder_config()
        config_autoencoder['project_id'] = config['project_id']
        config_autoencoder['video_id'] = config['video_id']
        config_autoencoder['project_name'] = config['project_name']
        config['autoencoder_model'] = autoencoder.train(config_autoencoder)
    print('[*] trained autoencoder model',config['autoencoder_model'])

    # 4) train keypoint estimator model
    if config['keypoint_model'] is None:
        config['max_steps'] = 50000
        model.create_train_dataset(config)
        config['keypoint_model'] = roi_segm.train(config)
    print('[*] trained keypoint_model',config['keypoint_model'])
    # </train models>

    # <load models>
    # load trained object detection model
    if detection_model is None:
        # load config json to know which backbone was used 
        with open(os.path.join(config['objectdetection_model'],'config.json')) as json_file:
            objconfig = json.load(json_file)
        detection_model = finetune.load_trained_model(objconfig)
    # load trained autoencoder model for Deep Sort Tracking 
    encoder_model = deep_sort_app.load_feature_extractor(config)

    # load trained keypoint model
    keypoint_model = load_keypoint_model(config['keypoint_model'])
    # </load models>

    # 3) run bbox tracking deep sort with fixed tracks
    min_confidence = 0.5 # Detection confidence threshold. Disregard all detections that have a confidence lower than this value.
    nms_max_overlap = 1.0 # Non-maxima suppression threshold: Maximum detection overlap
    max_cosine_distance = 0.2 # Gating threshold for cosine distance metric (object appearance).
    nn_budget = None # Maximum size of the appearance descriptors gallery. If None, no budget is enforced.
    display = True # dont write vis images

    print(config)
    crop_dim = roi_segm.get_roi_crop_dim(config['project_id'], config['video_id'], cv.imread(frame_files[0]).shape[0])
    deep_sort_app.run(config, detection_model, encoder_model, keypoint_model, output_dir, 
            args.min_confidence_boxes, args.min_confidence_keypoints, crop_dim, nms_max_overlap, max_cosine_distance, nn_budget, display)
    
    video_file = os.path.join(video.get_project_dir(video.base_dir_default, config['project_id']),'tracking_%s_vis%i.mp4' % (config['project_name'],config['video_id']))
    
    convert_video_h265(video_file.replace('.mp4','.avi'), video_file)
    print('[*] done tracking')
    
def convert_video_h265(video_in, video_out):
    import subprocess 
    if os.path.isfile(video_out):
        os.remove(video_out)
    subprocess.call(['ffmpeg','-i',video_in, '-c:v','libx265','-preset','ultrafast',video_out])
    os.remove(video_in)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--objectdetection_model', required=False,default=None)
    parser.add_argument('--keypoint_model', required=False,default=None)
    parser.add_argument('--autoencoder_model', required=False,default=None)
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    parser.add_argument('--train_video_ids',default='')
    parser.add_argument('--minutes',required=False,default=0.0,type=float)
    parser.add_argument('--min_confidence_boxes',required=False,default=0.5,type=float)
    parser.add_argument('--min_confidence_keypoints',required=False,default=0.5,type=float)
    #parser.add_argument('--fixed_number',required=False,default=4,type=int)
    args = parser.parse_args()
    
    main(args)