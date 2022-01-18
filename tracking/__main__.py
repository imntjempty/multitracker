
"""
    main program to track animals and their corresponding limbs on a video file

    
    python3.7 -m multitracker.tracking --project_id 1 --train_video_ids 1 --test_video_ids 1 --upper_bound 4 --video /home/alex/data/multitracker/projects/1/videos/2020-11-25_08-47-15_22772819_rec-00.00.00.000-00.10.20.916-seg1.avi --keypoint_method none --objectdetection_model /home/alex/github/multitracker/object_detection/YOLOX/YOLOX_outputs/yolox_voc_m/last_epoch_ckpt.pth 
             --sketch_file /home/alex/data/multitracker/projects/7/13/sketch.png 


    
"""

from tqdm import tqdm 
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
from collections import deque 

from natsort import natsorted
from multitracker import util 
from multitracker.be import video
from multitracker.keypoint_detection import heatmap_drawing, model , roi_segm
from multitracker.object_detection import finetune
from multitracker.tracking import inference 

from multitracker import autoencoder

from multitracker.tracking.viou import viou_tracker
from multitracker.tracking import upperbound_tracker
from multitracker.tracking.deep_sort.deep_sort import tracker as deepsort_tracker

from multitracker.tracking.deep_sort import deep_sort_app
from multitracker.tracking.keypoint_tracking import tracker as keypoint_tracking
from multitracker.tracking.deep_sort.application_util import preprocessing
from multitracker.tracking.deep_sort.application_util import visualization

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
    config['yolox_exp'] = args.yolox_exp
    config['yolox_name'] = args.yolox_name
    config['min_confidence_boxes'] = args.min_confidence_boxes
    
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
    if 'objectdetection_model' in config and config['objectdetection_model'] is not None:
        detection_model = inference.load_object_detector(config)

    else:
        config['objectdetection_max_steps'] = 30000
        # train object detector
        now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
        checkpoint_directory_object_detection = os.path.expanduser('~/checkpoints/multitracker/bbox/vids%s-%s' % (config['train_video_ids'], now))
        object_detect_restore = None 
        detection_model = None
        #detection_model = finetune.finetune(config, checkpoint_directory_object_detection, checkpoint_restore = object_detect_restore)
        #print('[*] trained object detection model',checkpoint_directory_object_detection)
        config['object_detection_model'] = checkpoint_directory_object_detection
        train_yolox()

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
    '''if detection_model is None:
        # load config json to know which backbone was used 
        with open(os.path.join(config['objectdetection_model'],'config.json')) as json_file:
            objconfig = json.load(json_file)
        objconfig['objectdetection_model'] = config['objectdetection_model']
        detection_model = finetune.load_trained_model(objconfig)'''
    
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

    
    print(4*'\n',config)
    
    ttrack_start = time.time()
    
    run(config, detection_model, encoder_model, keypoint_model, args.min_confidence_boxes, args.min_confidence_keypoints  )
    
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


  
def run(config, detection_model, encoder_model, keypoint_model, min_confidence_boxes, min_confidence_keypoints, tracker = None):
    if 'UpperBound' == config['tracking_method']:
        assert 'upper_bound' in config and config['upper_bound'] is not None and int(config['upper_bound'])>0, "ERROR: Upper Bound Tracking requires the argument --upper_bound to bet set (eg --upper_bound 4)"
    #config['upper_bound'] = None # ---> force VIOU tracker
    video_reader = cv.VideoCapture( config['video'] )
    
    Wframe  = int(video_reader.get(cv.CAP_PROP_FRAME_WIDTH))
    Hframe = int(video_reader.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    crop_dim = roi_segm.get_roi_crop_dim(config['data_dir'], config['project_id'], config['test_video_ids'].split(',')[0],Hframe)
    total_frame_number = int(video_reader.get(cv.CAP_PROP_FRAME_COUNT))
    print('[*] total_frame_number',total_frame_number,'Hframe,Wframe',Hframe,Wframe,'crop_dim',crop_dim)
    
    video_file_out = inference.get_video_output_filepath(config)
    if config['file_tracking_results'] is None:
        config['file_tracking_results'] = video_file_out.replace('.%s'%video_file_out.split('.')[-1],'.csv')
    # setup CSV for object tracking and keypoints
    print('[*] writing csv file to', config['file_tracking_results'])
    file_csv = open( config['file_tracking_results'], 'w') 
    file_csv.write('video_id,frame_id,track_id,center_x,center_y,x1,y1,x2,y2,time_since_update\n')
    if 'keypoint_method' in config and not config['keypoint_method'] == 'none':
        file_csv_keypoints = open( config['file_tracking_results'].replace('.csv','_keypoints.csv'), 'w') 
        file_csv_keypoints.write('video_id,frame_id,keypoint_class,keypoint_x,keypoint_y\n')
    

    # find out if video is part of the db and has video_id
    try:
        db.execute("select id from videos where name == '%s'" % config['video'].split('/')[-1])
        video_id = int([x for x in db.cur.fetchall()][0])
    except:
        video_id = -1
    print('      video_id',video_id)

    if os.path.isfile(video_file_out): os.remove(video_file_out)
    import skvideo.io
    video_writer = skvideo.io.FFmpegWriter(video_file_out, outputdict={
        '-vcodec': 'libx264',  #use the h.264 codec
        '-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }) 

    visualizer = visualization.Visualization([Wframe, Hframe], update_ms=5, config=config)
    print('[*] writing video file %s' % video_file_out)
    
    ## initialize tracker for boxes and keypoints
    if config['tracking_method'] == 'UpperBound':
        tracker = upperbound_tracker.UpperBoundTracker(config)
    elif config['tracking_method'] == 'DeepSORT':
        tracker = deepsort_tracker.DeepSORTTracker(config)
    elif config['tracking_method'] == 'VIoU':
        tracker = viou_tracker.VIoUTracker(config)
    keypoint_tracker = keypoint_tracking.KeypointTracker()

    frame_idx = -1
    frame_buffer = deque()
    detection_buffer = deque()
    keypoint_buffer = deque()
    results = []
    running = True 
    scale = None 
    
    tbenchstart = time.time()
    # fill up initial frame buffer for batch inference
    for ib in range(config['inference_objectdetection_batchsize']-1):
        ret, frame = video_reader.read()
        #frame = cv.resize(frame,None,fx=0.5,fy=0.5)
        frame_buffer.append(frame[:,:,::-1]) # trained on TF RGB, cv2 yields BGR

    #while running: #video_reader.isOpened():
    for frame_idx in tqdm(range(total_frame_number)):
        #frame_idx += 1 
        config['count'] = frame_idx
        if frame_idx == 10:
            tbenchstart = time.time()

        # fill up frame buffer as you take something from it to reduce lag 
        timread0 = time.time()
        if video_reader.isOpened():
            ret, frame = video_reader.read()
            #frame = cv.resize(frame,None,fx=0.5,fy=0.5)
            if frame is not None:
                frame_buffer.append(frame[:,:,::-1]) # trained on TF RGB, cv2 yields BGR
            else:
                running = False
                file_csv.close()
                if 'keypoint_method' in config and not config['keypoint_method'] == 'none':
                    file_csv_keypoints.close()
                return True  
        else:
            running = False 
            file_csv.close()
            if 'keypoint_method' in config and not config['keypoint_method'] == 'none':
                file_csv_keypoints.close()
            return True 
        timread1 = time.time()
        showing = True # frame_idx % 1000 == 0

        if running:
            tobdet0 = time.time()
            if len(detection_buffer) == 0:
                frames_tensor = np.array(list(frame_buffer)).astype(np.float32)
                # fill up frame buffer and then detect boxes for complete frame buffer
                t_odet_inf_start = time.time()
                batch_detections = inference.detect_batch_bounding_boxes(config, detection_model, frames_tensor, min_confidence_boxes)
                [detection_buffer.append(batch_detections[ib]) for ib in range(config['inference_objectdetection_batchsize'])]
                t_odet_inf_end = time.time()
                if frame_idx < 100 and frame_idx % 10 == 0:
                    print('  object detection ms',(t_odet_inf_end-t_odet_inf_start)*1000.,"batch", len(batch_detections),len(detection_buffer), (t_odet_inf_end-t_odet_inf_start)*1000./len(batch_detections) ) #   roughly 70ms

                if keypoint_model is not None:
                    t_kp_inf_start = time.time()
                    keypoint_buffer = inference.inference_batch_keypoints(config, keypoint_model, crop_dim, frames_tensor, detection_buffer, min_confidence_keypoints)
                    #[keypoint_buffer.append(batch_keypoints[ib]) for ib in range(config['inference_objectdetection_batchsize'])]
                    t_kp_inf_end = time.time()
                    if frame_idx < 200 and frame_idx % 10 == 0:
                        print('  keypoint ms',(t_kp_inf_end-t_kp_inf_start)*1000.,"batch", len(keypoint_buffer),(t_kp_inf_end-t_kp_inf_start)*1000./ (1e-6+len(keypoint_buffer)) ) #   roughly 70ms
            tobdet1 = time.time()

            # if detection buffer not empty use preloaded frames and preloaded detections
            frame = frame_buffer.popleft()
            detections = detection_buffer.popleft()
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            features = np.array([d.feature for d in detections])
            tobtrack0 = time.time()
            # Update tracker
            tracker.step({'img':frame,'detections':[detections, boxes, scores, features], 'frame_idx': frame_idx})
            tobtrack1 = time.time()
            tkptrack0 = time.time()
            if keypoint_model is not None:
                keypoints = keypoint_buffer.popleft()
                # update tracked keypoints with new detections
                tracked_keypoints = keypoint_tracker.update(keypoints)
            else:
                keypoints = tracked_keypoints = []
            tkptrack1 = time.time()

            # Store results.        
            for track in tracker.tracks:
                bbox = track.to_tlwh()
                center0, center1, _, _ = util.tlhw2chw(bbox)
                _unmatched_steps = -1
                if hasattr(track,'time_since_update'):
                    _unmatched_steps = track.time_since_update
                elif hasattr(track,'steps_unmatched'):
                    _unmatched_steps = track.steps_unmatched
                else:
                    raise Exception("ERROR: can't find track's attributes time_since_update or unmatched_steps")

                result = [video_id, frame_idx, track.track_id, center0, center1, bbox[0], bbox[1], bbox[2], bbox[3], _unmatched_steps]
                file_csv.write(','.join([str(r) for r in result])+'\n')
                results.append(result)
            
            if 'keypoint_method' in config and not config['keypoint_method'] == 'none':
                results_keypoints = []
                for kp in tracked_keypoints:
                    try:
                        result_keypoint = [video_id, frame_idx, kp.history_class[-1],kp.position[0],kp.position[1]]
                        file_csv_keypoints.write(','.join([str(r) for r in result_keypoint])+'\n')
                        results_keypoints.append(result_keypoint)
                    except Exception as e:
                        print(e)
            
            #print('[%i/%i] - %i detections. %i keypoints' % (config['count'], total_frame_number, len(detections), len(keypoints)))
            tvis0 = time.time()
            if showing:
                out = deep_sort_app.visualize(visualizer, frame, tracker, detections, keypoint_tracker, keypoints, tracked_keypoints, crop_dim, results, sketch_file=config['sketch_file'])
                video_writer.writeFrame(cv.cvtColor(out, cv.COLOR_BGR2RGB))
            tvis1 = time.time()

            if int(frame_idx) == 1010:
                tbenchend = time.time()
                print('[*] 1000 steps took',tbenchend-tbenchstart,'seconds')
                step_dur_ms = 1000.*(tbenchend-tbenchstart)/1000.
                fps = 1. / ( (tbenchend-tbenchstart)/1000. )
                print('[*] one time step takes on average',step_dur_ms,'ms',fps,'fps')

            if showing:
                cv.imshow("tracking visualization", cv.resize(out,None,None,fx=0.75,fy=0.75))
                cv.waitKey(1)
        
            if 0:
                dur_imread = timread1 - timread0
                dur_obdet = tobdet1 - tobdet0
                dur_obtrack = tobtrack1 - tobtrack0
                dur_kptrack = tkptrack1 - tkptrack0
                dur_vis = tvis1 - tvis0
                dur_imread, dur_obdet, dur_obtrack, dur_kptrack, dur_vis = [1000. * dur for dur in [dur_imread, dur_obdet, dur_obtrack, dur_kptrack, dur_vis]]
                print('imread',dur_imread,'obdetect',dur_obdet, 'obtrack',dur_obtrack, 'kptrack',dur_kptrack, 'vis',dur_vis)

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
    parser.add_argument('--min_confidence_boxes', required=False,default=0.5,type=float)
    parser.add_argument('--min_confidence_keypoints', required=False,default=0.5,type=float)
    parser.add_argument('--inference_objectdetection_batchsize', required=False,default=0,type=int)
    parser.add_argument('--inference_keypoint_batchsize', required=False,default=0,type=int)
    parser.add_argument('--output_tracking_results', required=False,default=None)
    parser.add_argument('--track_tail', required=False,default=100,type=int,help="How many steps back in the past should the path of each animal be drawn? -1 -> draw complete path")
    parser.add_argument('--sketch_file', required=False,default=None, help="Black and White Sketch of the frame without animals")
    parser.add_argument('--video', required=False,default=None)
    parser.add_argument('--yolox_exp', default='~/github/multitracker/object_detection/YOLOX/exps/example/yolox_voc/yolox_voc_m.py')
    parser.add_argument('--yolox_name', default='yolox_m')
    parser.add_argument('--tracking_method', required=False,default='UpperBound',type=str,help="Tracking Algorithm to use: [DeepSORT, VIoU, UpperBound] defaults to VIoU")
    parser.add_argument('--objectdetection_method', required=False,default="fasterrcnn", help="Object Detection Algorithm to use [fasterrcnn, ssd] defaults to fasterrcnn") 
    parser.add_argument('--objectdetection_resolution', required=False, default="640x640", help="xy resolution for object detection. coco pretrained model only available for 320x320, but smaller resolution saves time")
    parser.add_argument('--keypoint_resolution', required=False, default="224x224",help="patch size to analzye keypoints of individual animals")
    parser.add_argument('--keypoint_method', required=False,default="psp", help="Keypoint Detection Algorithm to use [none, hourglass2, hourglass4, hourglass8, vgg16, efficientnet, efficientnetLarge, psp]. defaults to psp") 
    parser.add_argument('--upper_bound', required=False,default=0,type=int)
    parser.add_argument('--data_dir', required=False, default = '~/data/multitracker')
    parser.add_argument('--delete_all_checkpoints', required=False, action="store_true")
    parser.add_argument('--video_resolution', default=None, help='resolution the video is downscaled to before processing to reduce runtime, eg 640x480. default no downscaling')
    parser.add_argument('--use_all_data4train', action='store_true')
    args = parser.parse_args()
    assert args.tracking_method in ['DeepSORT', 'VIoU', 'UpperBound']
    assert args.objectdetection_method in ['fasterrcnn', 'ssd']
    assert args.keypoint_method in ['none', 'hourglass2', 'hourglass4', 'hourglass8', 'vgg16', 'efficientnet', 'efficientnetLarge', 'psp']
    args.yolox_exp = os.path.expanduser(args.yolox_exp)
    args.data_dir = os.path.expanduser(args.data_dir)
    main(args)