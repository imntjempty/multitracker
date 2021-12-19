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
from numpy.core.defchararray import _strip_dispatcher
from tqdm import tqdm
import tensorflow as tf 
assert tf.__version__.startswith('2.'), 'YOU MUST INSTALL TENSORFLOW 2.X'
print('[*] TF version',tf.__version__)
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import Model

from multitracker.tracking.deep_sort.application_util import preprocessing
from multitracker.tracking.deep_sort.application_util import visualization

from multitracker.tracking.deep_sort.deep_sort.detection import Detection
#from multitracker.tracking.deep_sort.deep_sort.tracker import Tracker
from multitracker.keypoint_detection import roi_segm
from multitracker.tracking import inference
from multitracker.tracking.keypoint_tracking import tracker as keypoint_tracking
from multitracker.util import tlhw2chw
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
count_ok, count_failed = 0, 0

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

def draw_label(im, pos=(5,5), label="", font_size = 18, font_name = "Roboto-Regular.ttf"):
    from PIL import ImageFont, ImageDraw, Image
    pil_im = Image.fromarray(im)

    draw = ImageDraw.Draw(pil_im)

    # Choose a font
    try:
        font = ImageFont.truetype(font_name, font_size)
    except:
        font = ImageFont.load_default()

    # Draw the text
    draw.text(pos, label, font=font)

    im = np.array(pil_im)
    return im 
    



    cntlabel = 'Frame: %i' % vis.frame_idx
    text_size = cv.getTextSize(cntlabel, cv.FONT_HERSHEY_PLAIN, vis.viewer.font_scale, vis.viewer.thickness)
    center = 5, 5 + text_size[0][1]
    print('frame',frame.shape,frame.dtype)
    frame = cv.putText(frame, cntlabel, center, cv.FONT_HERSHEY_PLAIN,
                vis.viewer.font_scale, (255, 255, 255), vis.viewer.thickness)
    return frame 



def visualize(vis, frame, tracker, detections, keypoint_tracker, keypoints, tracked_keypoints, crop_dim, results, sketch_file = None):
    # draw keypoint detections 'whitish'
    im = np.array(frame, copy=True)
    font_size = int(24. * min(im.shape[:2])/1000.)
    
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
        vis_crop = np.array(im[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:],copy=True)

        # make track id sensitive transparent background header
        header_height_px = int(1.2 * font_size)
        overlayed_header = vis_crop[:header_height_px,:,:]
        _alpha = 0.7
        track_color = tuple(visualization.create_unique_color_uchar(track.track_id))
        overlayed_header = np.uint8(overlayed_header*_alpha + track_color * np.ones_like(overlayed_header) * (1. - _alpha))
        vis_crop[:overlayed_header.shape[0],:,:] = overlayed_header

        # draw status attributes in corner (totaltraveled_px, speed_px)
        if hasattr(track, 'speed_px') and track.speed_px is not None and hasattr(track, 'totaltraveled_px') and track.totaltraveled_px is not None:
            str_speed = str(int(track.speed_px))+"."+str(track.speed_px-int(track.speed_px))[2:5]
            vis_crop = draw_label(vis_crop, pos = (5,5), label = "distance traveled (px): %i\ncurrent speed (px/sec): %s" % (track.totaltraveled_px, str_speed), font_size=font_size)

        vis_crops.append( vis_crop )

    for i, track in enumerate(tracker.tracks):    
        # draw visualization of this tracker complete history
        vis_history_track = np.zeros(frame.shape,'uint8')
        if hasattr(track, 'last_means'):
            for j in range(1,len(track.last_means)):
                p = tuple([int(round(cc)) for cc in track.last_means[j-1][:2]]) # centerx, centery, width, height
                q = tuple([int(round(cc)) for cc in track.last_means[j  ][:2]])
                vis_history_track = cv.line(vis_history_track, p, q, tuple(visualization.create_unique_color_uchar(track.track_id)),3) 
        elif hasattr(track, 'history'):
            for j in range(1,len(track.history)):
                pb = tuple([int(round(cc)) for cc in track.history[j-1]['bbox']]) # left,top,width,height
                qb = tuple([int(round(cc)) for cc in track.history[j  ]['bbox']])
                p = (pb[0]+pb[2]//2,pb[1]+pb[3]//2)
                q = (qb[0]+qb[2]//2,qb[1]+qb[3]//2)
                vis_history_track = cv.line(vis_history_track, p, q, tuple(visualization.create_unique_color_uchar(track.track_id)),3) 

        vis_crops.append(cv.resize(vis_history_track,(256,256)))
    
    _shape = [im.shape[0]//2,im.shape[1]//2]
    if len(vis_crops)>0:
        _shape = vis_crops[0].shape[:2]
    
    ## draw frame counter for evaluation
    #frame = draw_label(frame, label= "Frame: %i" % vis.frame_idx)
    ## draw trackers and detections
    vis.set_image(frame.copy())
    vis.draw_detections(detections)
    vis.draw_trackers(tracker.tracks)
    
    out = append_crop_mosaic(vis.viewer.image,vis_crops)

    if 0:
        # scale down visualization
        max_height = 600
        ratio = max_height / out.shape[0]
        out = cv.resize(out, None, fx=ratio,fy=ratio)
    return out 
