
"""
    main program to track animals and their corresponding limbs on a video file

    
    python3.7 -m multitracker.tracking --project_id 7 --train_video_ids 9,14 --test_video_ids 13,14 --objectdetection_model /home/alex/checkpoints/experiments/MiceTop/E/1-2020-12-20_18-23-08 --keypoint_model /home/alex/checkpoints/experiments/MiceTop/B/random-2020-12-20_19-04-50 --min_confidence_boxes 0.85 --tracking_method UpperBound --upper_bound 4 --video /home/alex/data/multitracker/projects/7/videos/from_above_Oct2020_2_12fps.mp4 --sketch_file /home/alex/data/multitracker/projects/7/13/sketch.png 
"""

import os
import numpy as np 
import tensorflow as tf 
tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel('ERROR')
import subprocess
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py
import json 
import shutil
from natsort import natsorted
from multitracker import util 
from multitracker.be import video
from multitracker.keypoint_detection import heatmap_drawing, model , roi_segm
from multitracker.object_detection import finetune
from multitracker.tracking import inference 
from multitracker.keypoint_detection.roi_segm import get_center
from multitracker import autoencoder

from multitracker.tracking.tracklets import get_tracklets
from multitracker.tracking.clustering import get_clustlets

from multitracker.tracking.deep_sort import deep_sort_app
from multitracker.tracking.viou import viou_tracker
from multitracker.tracking import upperbound_tracker


def main(args):
    os.environ['MULTITRACKER_DATA_DIR'] = args.data_dir
    from multitracker.be import dbconnection
    
    if args.minutes>0 or args.video_resolution is not None:
        tpreprocessvideostart = time.time()
        sscale = args.video_resolution if args.video_resolution is not None else ''
        smins = str(args.minutes) if args.minutes>0 else ''
        fvo = args.video[:-4] + sscale + smins + args.video[-4:]
        if not os.path.isfile(fvo):
            commands = ['ffmpeg']
            if args.minutes>0:
                commands.extend(['-t',str(int(60.*args.minutes))])
            commands.extend(['-i',args.video])
            if args.video_resolution is not None:
                commands.extend(['-vf','scale=%s'%args.video_resolution.replace('x',':')])
            commands.extend([fvo])
            print('[*] preprocess video', ' '.join(commands))
            subprocess.call(commands)
            tpreprocessvideoend = time.time()
            print('[*] preprocessing of video to %s took %f seconds' % (fvo, tpreprocessvideoend-tpreprocessvideostart))
        args.video = fvo

    tstart = time.time()
    config = model.get_config(project_id = args.project_id)
    config['project_id'] = args.project_id
    config['video'] = args.video
    config['keypoint_model'] = args.keypoint_model
    config['autoencoder_model'] = args.autoencoder_model 
    config['objectdetection_model'] = args.objectdetection_model
    config['train_video_ids'] = args.train_video_ids
    config['test_video_ids'] = args.test_video_ids
    config['minutes'] = args.minutes
    #config['upper_bound'] = db.get_video_fixednumber(args.video_id) 
    #config['upper_bound'] = None
    config['upper_bound'] = args.upper_bound
    config['n_blocks'] = 4
    config['tracking_method'] = args.tracking_method
    config['track_tail'] = args.track_tail
    config['sketch_file'] = args.sketch_file
    config['file_tracking_results'] = args.output_tracking_results
    config['use_all_data4train'] = args.use_all_data4train
    
    config['object_detection_backbone'] = args.objectdetection_method
    config = model.update_config_object_detection(config)
    config['object_detection_resolution'] = [int(r) for r in args.objectdetection_resolution.split('x')]
    config['keypoint_resolution'] = [int(r) for r in args.keypoint_resolution.split('x')]
    config['img_height'], config['img_width'] = config['keypoint_resolution'][::-1]
    config['kp_backbone'] = args.keypoint_method
    if 'hourglass' in args.keypoint_method:
        config['kp_num_hourglass'] = int(args.keypoint_method[9:])
        config['kp_backbone'] = 'efficientnetLarge'
    
    if args.inference_objectdetection_batchsize > 0:
        config['inference_objectdetection_batchsize'] = args.inference_objectdetection_batchsize
    if args.inference_keypoint_batchsize > 0:
        config['inference_keypoint_batchsize'] = args.inference_keypoint_batchsize

    if args.delete_all_checkpoints:
        if os.path.isdir(os.path.expanduser('~/checkpoints/multitracker')):
            shutil.rmtree(os.path.expanduser('~/checkpoints/multitracker'))
    if args.data_dir:
        db = dbconnection.DatabaseConnection(file_db=os.path.join(args.data_dir,'data.db'))
        config['data_dir'] = args.data_dir 
        config['kp_data_dir'] = os.path.join(args.data_dir , 'projects/%i/data' % config['project_id'])
        config['kp_roi_dir'] = os.path.join(args.data_dir , 'projects/%i/data_roi' % config['project_id'])
        config['keypoint_names'] = db.get_keypoint_names(config['project_id'])
        config['project_name'] = db.get_project_name(config['project_id'])

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
    if config['autoencoder_model'] is None and config['tracking_method'] == 'DeepSORT':
        config_autoencoder = autoencoder.get_autoencoder_config()
        config_autoencoder['project_id'] = config['project_id']
        config_autoencoder['video_ids'] = natsorted(list(set([int(iid) for iid in config['train_video_ids'].split(',')]+[int(iid) for iid in config['test_video_ids'].split(',')])))
        config_autoencoder['project_name'] = config['project_name']
        config_autoencoder['data_dir'] = config['data_dir']
        config['autoencoder_model'] = autoencoder.train(config_autoencoder)
    print('[*] trained autoencoder model',config['autoencoder_model'])

    # 4) train keypoint estimator model
    if config['keypoint_model'] is None and not config['kp_backbone'] == 'none':
        config['kp_max_steps'] = 25000
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
    if config['kp_backbone'] == 'none':
        keypoint_model = None
    else:
        keypoint_model = inference.load_keypoint_model(config['keypoint_model'])
    # </load models>

    # 3) run bbox tracking deep sort with fixed tracks
    nms_max_overlap = 1.0 # Non-maxima suppression threshold: Maximum detection overlap
    max_cosine_distance = 0.2 # Gating threshold for cosine distance metric (object appearance).
    nn_budget = None # Maximum size of the appearance descriptors gallery. If None, no budget is enforced.
    print(4*'\n',config)
    
    ttrack_start = time.time()
    if config['tracking_method'] == 'DeepSORT':
        deep_sort_app.run(config, detection_model, encoder_model, keypoint_model,  
            args.min_confidence_boxes, args.min_confidence_keypoints, nms_max_overlap, max_cosine_distance, nn_budget)
    elif config['tracking_method'] == 'UpperBound':
        upperbound_tracker.run(config, detection_model, encoder_model, keypoint_model, args.min_confidence_boxes, args.min_confidence_keypoints  )
    elif config['tracking_method'] == 'VIoU':
        viou_tracker.run(config, detection_model, encoder_model, keypoint_model, args.min_confidence_boxes, args.min_confidence_keypoints  )
    ttrack_end = time.time()
    ugly_big_video_file_out = inference.get_video_output_filepath(config)
    video_file_out = ugly_big_video_file_out.replace('.avi','.mp4')
    convert_video_h265(ugly_big_video_file_out, video_file_out)
    print('[*] done tracking after %f minutes. outputting file' % float(int((ttrack_end-ttrack_start)*10.)/10.),video_file_out)
    
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
    parser.add_argument('--project_id', required=True,type=int)
    parser.add_argument('--train_video_ids', default='')
    parser.add_argument('--test_video_ids', default='')
    parser.add_argument('--minutes', required=False,default=0.0,type=float,help="cut the video to the first n minutes, eg 2.5 cuts after the first 150seconds.")
    parser.add_argument('--min_confidence_boxes', required=False,default=0.65,type=float)
    parser.add_argument('--min_confidence_keypoints', required=False,default=0.5,type=float)
    parser.add_argument('--inference_objectdetection_batchsize', required=False,default=0,type=int)
    parser.add_argument('--inference_keypoint_batchsize', required=False,default=0,type=int)
    parser.add_argument('--output_tracking_results', required=False,default=None)
    parser.add_argument('--track_tail', required=False,default=100,type=int,help="How many steps back in the past should the path of each animal be drawn? -1 -> draw complete path")
    parser.add_argument('--sketch_file', required=False,default=None, help="Black and White Sketch of the frame without animals")
    parser.add_argument('--video', required=False,default=None)
    parser.add_argument('--tracking_method', required=False,default='UpperBound',type=str,help="Tracking Algorithm to use: [DeepSORT, VIoU, UpperBound] defaults to VIoU")
    parser.add_argument('--objectdetection_method', required=False,default="fasterrcnn", help="Object Detection Algorithm to use [fasterrcnn, ssd] defaults to fasterrcnn") 
    parser.add_argument('--objectdetection_resolution', required=False, default="640x640", help="xy resolution for object detection. coco pretrained model only available for 640x640, but smaller resolution saves time")
    parser.add_argument('--keypoint_resolution', required=False, default="224x224",help="patch size to analzye keypoints of individual animals")
    parser.add_argument('--keypoint_method', required=False,default="vgg16", help="Keypoint Detection Algorithm to use [none, hourglass2, hourglass4, hourglass8, vgg16, efficientnet, efficientnetLarge, psp]. defaults to hourglass2") 
    parser.add_argument('--upper_bound', required=False,default=0,type=int)
    parser.add_argument('--data_dir', required=False, default = os.path.expanduser('~/data/multitracker'))
    parser.add_argument('--delete_all_checkpoints', required=False, action="store_true")
    parser.add_argument('--video_resolution', default=None, help='resolution the video is downscaled to before processing to reduce runtime, eg 640x480. default no downscaling')
    parser.add_argument('--use_all_data4train', action='store_true')
    args = parser.parse_args()
    assert args.tracking_method in ['DeepSORT', 'VIoU', 'UpperBound']
    assert args.objectdetection_method in ['fasterrcnn', 'ssd']
    assert args.keypoint_method in ['none', 'hourglass2', 'hourglass4', 'hourglass8', 'vgg16', 'efficientnet', 'efficientnetLarge', 'psp']
    main(args)