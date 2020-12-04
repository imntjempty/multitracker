
"""
    main program to track animals and their corresponding limbs on a video file

    python3.7 -m multitracker.tracking --project_id 7 --video_id 13 --train_video_ids 9,14 --objectdetection_model ~/checkpoints/multitracker/bbox/vids9\,14-2020-11-13_07-56-02 --autoencoder_model /home/alex/checkpoints/multitracker/ae/vid_13-2020-11-10_22-34-32 --keypoint_model /home/alex/checkpoints/multitracker/keypoints/vids9,14-2020-11-13_00-24-28 --min_confidence_boxes 0.6 --min_confidence_keypoints 0.7 --tracking_method Tracktor --video /home/alex/data/multitracker/projects/7/videos/from_above_Oct2020_2_12fps.mp4

    python3.7 -m multitracker.tracking --project_id 7 --video_id 13 --train_video_ids 9,14 --objectdetection_model ~/checkpoints/multitracker/bbox/vids9\,14-2020-11-13_07-56-02 --autoencoder_model /home/alex/checkpoints/multitracker/ae/vid_13-2020-11-10_22-34-32 --keypoint_model /home/alex/checkpoints/multitracker/keypoints/vids9,14-2020-11-13_00-24-28 --min_confidence_boxes 0.7 --min_confidence_keypoints 0.5 --tracking_method FixedAssigner --video /home/alex/data/multitracker/projects/7/videos/from_above_Oct2020_2_12fps.mp4


    python3.7 -m multitracker.tracking --project_id 7 --video_id 13 --train_video_ids 9,14 --autoencoder_model /home/alex/checkpoints/multitracker/ae/vid_13-2020-11-10_22-34-32 --keypoint_model /home/alex/checkpoints/multitracker/keypoints/vids9,14-2020-11-13_00-24-28 --min_confidence_boxes 0.5 --min_confidence_keypoints 0.5 --tracking_method FixedAssigner --video /home/alex/data/multitracker/projects/7/videos/from_above_Oct2020_2_12fps.mp4 --objectdetection_model /home/alex/checkpoints/multitracker/bbox/flips,rot90,gauss,noise-vids9,14-2020-11-25_07-15-17 --sketch_file /home/alex/data/multitracker/projects/7/13/sketch.png

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
from multitracker.object_detection import finetune
from multitracker.tracking import inference 
from multitracker.keypoint_detection.roi_segm import get_center
from multitracker import autoencoder

from multitracker.tracking.tracklets import get_tracklets
from multitracker.tracking.clustering import get_clustlets

from multitracker.tracking.deep_sort import deep_sort_app
from multitracker.tracking.tracktor import tracktor_app
from multitracker.tracking import cv_multitracker

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

def main(args):
    tstart = time.time()
    config = model.get_config(project_id = args.project_id)
    config['project_id'] = args.project_id
    config['video_id'] = args.video_id
    config['video'] = args.video
    config['keypoint_model'] = args.keypoint_model
    config['autoencoder_model'] = args.autoencoder_model 
    config['objectdetection_model'] = args.objectdetection_model
    config['train_video_ids'] = args.train_video_ids
    config['minutes'] = args.minutes
    #config['fixed_number'] = db.get_video_fixednumber(args.video_id) 
    #config['fixed_number'] = None
    config['fixed_number'] = args.fixed_number
    config['n_blocks'] = 4
    if args.inference_objectdetection_batchsize > 0:
        config['inference_objectdetection_batchsize'] = args.inference_objectdetection_batchsize
    config['tracking_method'] = args.tracking_method
    config['track_tail'] = args.track_tail
    config['sketch_file'] = args.sketch_file
    config['file_tracking_results'] = args.output_tracking_results
    
    config['object_detection_backbone'] = args.objectdetection_method
    config = model.update_config_object_detection(config)
    config['backbone'] = args.keypoint_method
    if 'hourglass' in args.keypoint_method:
        config['num_hourglass'] = int(args.keypoint_method[9:])
        config['backbone'] = 'efficientnetLarge'
    
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
        objconfig['objectdetection_model'] = config['objectdetection_model']
        detection_model = finetune.load_trained_model(objconfig)
    
    # load trained autoencoder model for Deep Sort Tracking 
    encoder_model = None 
    if config['tracking_method']=='DeepSORT':
        encoder_model = inference.load_autoencoder_feature_extractor(config)

    # load trained keypoint model
    keypoint_model = inference.load_keypoint_model(config['keypoint_model'])
    # </load models>

    # 3) run bbox tracking deep sort with fixed tracks
    nms_max_overlap = 1.0 # Non-maxima suppression threshold: Maximum detection overlap
    max_cosine_distance = 0.2 # Gating threshold for cosine distance metric (object appearance).
    nn_budget = None # Maximum size of the appearance descriptors gallery. If None, no budget is enforced.
    display = True # dont write vis images

    crop_dim = roi_segm.get_roi_crop_dim(config['project_id'], config['video_id'], cv.imread(frame_files[0]).shape[0])
    if config['tracking_method'] == 'DeepSORT':
        deep_sort_app.run(config, detection_model, encoder_model, keypoint_model, output_dir, 
            args.min_confidence_boxes, args.min_confidence_keypoints, crop_dim, nms_max_overlap, max_cosine_distance, nn_budget, display)
    else:
        # tracktor algorithm
        if config['tracking_method'] == 'Tracktor':
            tracktor_app.run(config, detection_model, encoder_model, keypoint_model, crop_dim)
        else:# config['tracking_method'] == 'FixedAssigner':
            cv_multitracker.run(config, detection_model, encoder_model, keypoint_model, crop_dim, args.min_confidence_boxes, args.min_confidence_keypoints  )

    video_file_out = inference.get_video_output_filepath(config)
    convert_video_h265(video_file_out.replace('.mp4','.avi'), video_file_out)
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
    parser.add_argument('--inference_objectdetection_batchsize',required=False,default=0,type=int)
    parser.add_argument('--output_tracking_results',required=False,default=None)
    parser.add_argument('--track_tail',required=False,default=800,type=int,help="How many steps back in the past should the path of each animal be drawn? -1 -> draw complete path")
    parser.add_argument('--sketch_file',required=False,default=None, help="Black and White Sketch of the frame without animals")
    parser.add_argument('--video',required=False,default=None)
    parser.add_argument('--tracking_method',required=False,default='DeepSORT',type=str,help="Tracking Algorithm to use: [DeepSORT, VIoU, FixedAssigner] defaults to DeepSORT")
    parser.add_argument('--objectdetection_method',required=False,default="fasterrcnn", help="Object Detection Algorithm to use [fasterrcnn, ssd] defaults to fasterrcnn") 
    parser.add_argument('--keypoint_method',required=False,default="hourglass2", help="Keypoint Detection Algorithm to use [hourglass2, hourglass4, hourglass8, vgg16, efficientnet, efficientnetLarge, psp]. defaults to hourglass2") 
    parser.add_argument('--fixed_number',required=False,default=0,type=int)
    args = parser.parse_args()
    assert args.tracking_method in ['DeepSORT', 'VIoU', 'FixedAssigner']
    assert args.objectdetection_method in ['fasterrcnn', 'ssd']
    assert args.keypoint_method in ['hourglass2', 'hourglass4', 'hourglass8', 'vgg16', 'efficientnet', 'efficientnetLarge', 'psp']
    main(args)