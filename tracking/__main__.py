
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
from multitracker.tracking.inference import load_data, load_model, inference_heatmap, get_heatmaps_keypoints
from multitracker.tracking.tracklets import get_tracklets
from multitracker.tracking.clustering import get_clusters
from multitracker.object_detection import finetune
from multitracker.tracking.deep_sort import deep_sort_app
from multitracker import autoencoder

def get_detections(config, model_path, project_id, video_id, max_minutes = 0, thresh_detection = 0.5):
    project_id = int(project_id)
    video_id = int(video_id)
    output_dir = os.path.expanduser('~/data/multitracker/tracks/%i/%i/%s' % (project_id, video_id, model_path.split('/')[-1].split('.')[0]))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # load pretrained model
    trained_model = load_model(model_path)

    # load frame files
    frame_files = load_data(project_id,video_id, max_minutes=max_minutes)
    config['input_image_shape'] = cv.imread(frame_files[0]).shape[:2]

    detections = {} 
    
    for i, frame_file in enumerate(frame_files):
        tframestart = time.time()
        fnameo = os.path.join(output_dir,frame_file.split('/')[-1])
        # inference frame
        #print('frame_file',frame_file)
        frame = cv.imread(frame_file)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        tread = time.time()
        heatmaps = inference_heatmap(config, trained_model, frame)
        tinference = time.time()
        # detect keypoints
        keypoints = get_heatmaps_keypoints(heatmaps,thresh_detection=thresh_detection)
        tkeypoints = time.time()
        detections[frame_file] = keypoints
        tframeend = time.time()
        eta_min = (len(frame_files)-i) * (tframeend - tframestart) / 60.
        print('[* %i/%i]'%(i,len(frame_files)), fnameo ,'heatmaps',heatmaps.shape,'min/max %f/%f'%(heatmaps.min(),heatmaps.max()),'found %i keypoints'%len(keypoints),'in %f seconds. estimated %f minutes remaining'%(tframeend - tframestart,eta_min))
        if 1:
            print('[* timings] dump detections: ioread %f, inference %f, keypoints %f'%(tread-tframestart,tinference-tread,tkeypoints-tinference))
    
    #if 1:
    #    util.make_video(output_dir, output_dir+'.mp4',"%05d.png")
    
    return detections 
 
def visualize_boxes_with_keypoints(tracked_boxes, tracked_keypoints, video_file):
    print('[*] visualize_boxes_with_keypoints')
    print('boxes:',list(tracked_boxes.keys()),'\n')
    print('keypoints:',list(tracked_keypoints.keys()),'\n')
    print('video_file',video_file)

def main(args):
    tstart = time.time()
    config = model.get_config(project_id = args.project_id)
    config['video_id'] = args.video_id
    config['keypoint_model'] = args.keypoint_model
    config['autoencoder_model'] = args.autoencoder_model 
    config['object_model'] = args.object_model

    
    # train keypoint estimator model
    if config['keypoint_model'] is None:
        config['max_steps'] = 150000
        model.create_train_dataset(config)
        config['keypoint_model'] = model.train(config)

    # detection stage
    file_keypoint_detections = '/tmp/keypoint_detections-%i-%f-%f.npz' % (args.video_id, args.minutes, args.thresh_detection)
    print('file_keypoint_detections',file_keypoint_detections)
    if not os.path.isfile(file_keypoint_detections):
        detections = get_detections(config, config['keypoint_model'], args.project_id, args.video_id, max_minutes=args.minutes, thresh_detection=args.thresh_detection)
        np.savez_compressed('.'.join(file_keypoint_detections.split('.')[:-1]), detections=detections)
        wrote_detections = True 
    else:
        detections = np.load(file_keypoint_detections, allow_pickle=True)['detections'].item()
        wrote_detections = False 

    # tracklet stage
    file_tracklets = '/tmp/tracklets-%i-%f-%f.npz' % (args.video_id, args.minutes, args.thresh_detection)
    print('file_tracklets',file_tracklets)
    if wrote_detections or not os.path.isfile(file_tracklets):
        tracklets = get_tracklets(detections, config['keypoint_model'], config, args.project_id, args.video_id)
        np.savez_compressed('.'.join(file_tracklets.split('.')[:-1]), tracklets=tracklets)
        wrote_tracklets = True 
    else:
        tracklets = np.load(file_tracklets, allow_pickle=True)['tracklets']#.item()
        wrote_tracklets = False 

    # tracklet clustering stage
    file_clusters = '/tmp/clusters-%i-%f-%f.npz' % (args.video_id, args.minutes, args.thresh_detection)
    print('file_clusters',file_clusters)
    if wrote_tracklets or not os.path.isfile(file_clusters):
        clusters = get_clusters(tracklets)
        np.savez_compressed('.'.join(file_clusters.split('.')[:-1]), clusters=clusters)
        wrote_clusters = True
    else:
        clusters = np.load(file_clusters, allow_pickle=True)['clusters']#.item()
        wrote_clusters = False
    
    # animal bounding box finetuning -> trains and inferences 
    config['max_steps'] = 15001
    detection_file_bboxes = '/tmp/multitracker/object_detection/predictions/%i/15000_bboxes_*.npz' % config['video_id']
    # train object detector
    if len(glob(detection_file_bboxes)) == 0:
        finetune.finetune(config)

    # run animal tracking
    detection_file_trackbedbboxes = '/tmp/multitracker/tracked_boxes_%i.npz' % config['video_id']

    # train autoencoder for tracking appearence vector
    if config['autoencoder_model'] is None:
        config['autoencoder_model'] = autoencoder.train()

    min_confidence = 0.8 # Detection confidence threshold. Disregard all detections that have a confidence lower than this value.
    nms_max_overlap = 1.0 # Non-maxima suppression threshold: Maximum detection overlap
    max_cosine_distance = 0.2 # Gating threshold for cosine distance metric (object appearance).
    nn_budget = None # Maximum size of the appearance descriptors gallery. If None, no budget is enforced.
    display = False # dont write vis images

    if not os.path.isfile(detection_file_trackbedbboxes):
        deep_sort_app.run(
            config, detection_file_bboxes, detection_file_trackbedbboxes,
            min_confidence, nms_max_overlap, min_detection_height,
            max_cosine_distance, nn_budget, display, feature_extractor_path)
    tracked_boxes = np.load(detection_file_trackbedbboxes, allow_pickle=True)['tracked_boxes']
    print('[*] loaded bbox animal tracking for %i frames' % len(list(tracked_boxes.keys())))

    # visualize merged results
    video_file = '/tmp/tracking_%i.mp4' % config['video_id']
    visualize_boxes_with_keypoints(tracked_boxes, clusters, video_file)

    tend = time.time()
    duration_min = int((tend - tstart)/60.)
    duration_h = int((tend - tstart)/3600.)
    print('[*] finished tracking after %i minutes. (%i hours)' % (duration_min, duration_h))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoint_model', required=False,default=None)
    parser.add_argument('--autoencoder_model', required=False,default=None)
    parser.add_argument('--object_model', required=False,default=None)
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    parser.add_argument('--minutes',required=False,default=0.0,type=float)
    parser.add_argument('--thresh_detection',required=False,default=0.5,type=float)
    args = parser.parse_args()
    main(args)