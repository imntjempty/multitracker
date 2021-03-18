# vim: expandtab:ts=4:sw=4
# python3.7 -m multitracker.tracking.deep_sort.deep_sort_app --detection_file '/tmp/multitracker/object_detection/predictions/9/60000_bboxes_*.npz' --project_id 7 --video_id 9

from __future__ import division, print_function, absolute_import

import argparse
import os
from glob import glob 
import subprocess
from collections import deque
import cv2 as cv 
import time 
import numpy as np
from tqdm import tqdm
import tensorflow as tf 
assert tf.__version__.startswith('2.'), 'YOU MUST INSTALL TENSORFLOW 2.X'
print('[*] TF version',tf.__version__)
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import Model

from multitracker.tracking.deep_sort.application_util import preprocessing
from multitracker.tracking.deep_sort.application_util import visualization
from multitracker.tracking.deep_sort.deep_sort import nn_matching
from multitracker.tracking.deep_sort.deep_sort.detection import Detection
from multitracker.tracking.deep_sort.deep_sort.tracker import Tracker
from multitracker.keypoint_detection import roi_segm
from multitracker.tracking import inference
from multitracker.tracking.keypoint_tracking import tracker as keypoint_tracking
from multitracker.tracking.upperbound_tracker import tlhw2chw
from multitracker.be import video
from multitracker import util 

colors = util.get_colors()

def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


global count_failed
global count_ok
count_ok, count_failed =0,0

def append_crop_mosaic(frame, vis_crops):
    # stack output image together
    # left side: original visualization with boxes and tracks
    # right side: moasics focused on tracks with keypoints
    if len(vis_crops) == 0:
        return np.hstack((frame,np.zeros((frame.shape[0],frame.shape[0],3),'uint8')))

    n = int(np.sqrt(len(vis_crops)))
    if not len(vis_crops) == int(n+0.5) ** 2: # is not square root?
        n += 1 
    n = max(2,n)

    d = int(frame.shape[0]/n)
    mosaic = np.zeros((frame.shape[0],frame.shape[0],3),'uint8')
    for i in range(len(vis_crops)):
        vis_crops[i] = cv.resize(vis_crops[i],(d,d))
        cx = i%n
        cy = i//n 
        mosaic[d*cy:d*cy+d,d*cx:d*cx+d,:] = vis_crops[i]

    out = np.hstack((frame, mosaic))
    return out

def draw_heatmap(frame, results, sketch_file):
    ##   results is list of [frame_idx, track.track_id, center0, center1, bbox[0], bbox[1], bbox[2], bbox[3], track.is_confirmed(), track.time_since_update]

    grid_num_y = 32
    grid_height = int(frame.shape[0]/grid_num_y)
    grid_num_x = int(frame.shape[1]/grid_height)
    history_heatmap = np.zeros((grid_num_y, grid_num_x),'float32')
    for result in results:
        c0,c1 = result[2], result[3]
        history_heatmap[min(int(c1/grid_height),grid_num_y-1)][min(int(c0/grid_height),grid_num_x-1)] += 1. 
    
    history_heatmap = (history_heatmap-history_heatmap.min()) / (history_heatmap.max()-history_heatmap.min()+1e-7)
    history_heatmap = np.uint8(np.around(255. * history_heatmap))
    history_heatmap = cv.resize(history_heatmap,(256,256))
    
    vis_history_heatmap = cv.applyColorMap(history_heatmap, cv.COLORMAP_JET)

    if sketch_file is not None:
        sketch = cv.imread(sketch_file)
        sketch = cv.cvtColor(sketch, cv.COLOR_BGR2GRAY)
        vis_history_heatmap[:,:,1] = 255 - cv.resize(sketch,(256,256))

    return vis_history_heatmap 

def visualize(vis, frame, tracker, detections, keypoint_tracker, keypoints, tracked_keypoints, crop_dim, results, sketch_file = None):
    # draw keypoint detections 'whitish'
    im = np.array(frame, copy=True)
    
    radius_keypoint = 5
    # draw history of keypoint track 
    for i in range( len(tracked_keypoints) ):
        median_class = int(np.median(np.array(tracked_keypoints[i].history_class)))
        color_keypoint = [int(ss) for ss in colors[median_class%len(colors)]]
        for j in range( 1, len(tracked_keypoints[i].history_estimated)):
            p1 = tuple(np.int32(np.around(tracked_keypoints[i].history_estimated[j])))
            p2 = tuple(np.int32(np.around(tracked_keypoints[i].history_estimated[j-1])))
            im = cv.line(im, p1, p2, color_keypoint, 2)

    history_heatmap = draw_heatmap(frame, results, sketch_file)
    # crop keypointed vis 
    vis_crops = [ history_heatmap ]
    for i, track in enumerate(tracker.tracks):
        # crop image around track center
        x1,y1,x2,y2 = track.to_tlbr()
        center = roi_segm.get_center(x1,y1,x2,y2, frame.shape[0],frame.shape[1], crop_dim)
        vis_crops.append( im[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:] )    

    for i, track in enumerate(tracker.tracks):    
        # draw visualization of this tracker complete history
        vis_history_track = np.zeros(frame.shape,'uint8')
        for j in range(1,len(track.last_means)):
            p = tuple([int(round(cc)) for cc in track.last_means[j-1][:2]])
            q = tuple([int(round(cc)) for cc in track.last_means[j  ][:2]])
            vis_history_track = cv.line(vis_history_track, p, q, tuple(visualization.create_unique_color_uchar(track.track_id)),3) 
        vis_crops.append(cv.resize(vis_history_track,(256,256)))
    
    _shape = [im.shape[0]//2,im.shape[1]//2]
    if len(vis_crops)>0:
        _shape = vis_crops[0].shape[:2]
    
    vis.set_image(frame.copy())
    vis.draw_detections(detections)
    vis.draw_trackers(tracker.tracks)
    
    out = append_crop_mosaic(vis.viewer.image,vis_crops)
    return out 

def run(config, detection_model, encoder_model, keypoint_model, min_confidence_boxes, min_confidence_keypoints,  
        nms_max_overlap, max_cosine_distance,nn_budget):
    
    max_age = 30 
    #max_age = 5
    config['count'] = 0 
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    tracker = Tracker(metric)
    
    results = []


    video_reader = cv.VideoCapture( config['video'] )
    # ignore first 5 frames
    for _ in range(5):
        ret, frame = video_reader.read()
    [Hframe,Wframe,_] = frame.shape
    visualizer = visualization.Visualization([Wframe, Hframe], update_ms=5, config=config)
    
    crop_dim = roi_segm.get_roi_crop_dim(config['data_dir'], config['project_id'], config['test_video_ids'].split(',')[0],Hframe)
    total_frame_number = int(video_reader.get(cv.CAP_PROP_FRAME_COUNT))
    print('total_frame_number',total_frame_number,'crop_dim',crop_dim)
    video_file_out = inference.get_video_output_filepath(config)
    if os.path.isfile(video_file_out): os.remove(video_file_out)
    import skvideo.io
    video_writer = skvideo.io.FFmpegWriter(video_file_out, outputdict={
        '-vcodec': 'libx264',  #use the h.264 codec
        '-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }) 
    print('[*] writing video file %s' % video_file_out)
    
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
        frame_buffer.append(frame[:,:,::-1]) # trained on TF RGB, cv2 yields BGR

    for frame_idx in tqdm(range(total_frame_number)):
        config['count'] = frame_idx
        if frame_idx == 10:
            tbenchstart = time.time()

        # fill up frame buffer as you take something from it to reduce lag 
        if video_reader.isOpened():
            ret, frame = video_reader.read()
            if frame is not None:
                frame_buffer.append(frame[:,:,::-1]) # trained on TF RGB, cv2 yields BGR
            else:
                running = False
                return True  
        else:
            running = False 
            return True 
        
        showing = True # frame_idx % 1000 == 0

    
        if len(detection_buffer) == 0:
            frames_tensor = np.array(list(frame_buffer)).astype(np.float32)
            # fill up frame buffer and then detect boxes for complete frame buffer
            t_odet_inf_start = time.time()
            batch_detections = inference.detect_batch_bounding_boxes(config, detection_model, frames_tensor, min_confidence_boxes, encoder_model)
            [detection_buffer.append(batch_detections[ib]) for ib in range(config['inference_objectdetection_batchsize'])]
            t_odet_inf_end = time.time()
            if frame_idx < 200 and frame_idx % 10 == 0:
                print('  object detection ms',(t_odet_inf_end-t_odet_inf_start)*1000.,"batch", len(batch_detections),len(detection_buffer), (t_odet_inf_end-t_odet_inf_start)*1000./len(batch_detections) ) #   roughly 70ms

            t_kp_inf_start = time.time()
            keypoint_buffer = inference.inference_batch_keypoints(config, keypoint_model, crop_dim, frames_tensor, detection_buffer, min_confidence_keypoints)
            #[keypoint_buffer.append(batch_keypoints[ib]) for ib in range(config['inference_objectdetection_batchsize'])]
            t_kp_inf_end = time.time()
            if frame_idx < 200 and frame_idx % 10 == 0:
                print('  keypoint ms',(t_kp_inf_end-t_kp_inf_start)*1000.,"batch", len(keypoint_buffer),(t_kp_inf_end-t_kp_inf_start)*1000./ (1e-6+len(keypoint_buffer)) ) #   roughly 70ms
        # if detection buffer not empty use preloaded frames and preloaded detections
        frame = frame_buffer.popleft()
        detections = detection_buffer.popleft()

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
        
        # update tracked keypoints with new detections
        keypoints = keypoint_buffer.popleft()
        tracked_keypoints = keypoint_tracker.update(keypoints)

        # Store results.
        for track in tracker.tracks:
            bbox = track.to_tlwh()
            center0, center1, _, _ = tlhw2chw(bbox)
            result = [frame_idx, track.track_id, center0, center1, bbox[0], bbox[1], bbox[2], bbox[3], track.is_confirmed(), track.time_since_update]
            results.append(result)
        
        # Update visualization.
        out = visualize(visualizer, frame, tracker, detections, keypoint_tracker, keypoints, tracked_keypoints, crop_dim, results)
        #cv.imshow("deepsortapp", out)
        
        # write visualization as frame to video
        try:
            video_writer.writeFrame(cv.cvtColor(out, cv.COLOR_BGR2RGB)) #out[:,:,::-1])
            #count_ok += 1
            config['count'] += 1
        except Exception as e:
            #count_failed += 1 
            print('[*] Video Writing for frame_idx %s failed. image shape'%frame_idx,out.shape)
            print(e)

    update_ms = 2
    visualizer = visualization.Visualization([Wframe, Hframe], update_ms, config)
    visualizer.run(frame_callback)
