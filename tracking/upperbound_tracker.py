"""
    implement OpenCV multi tracker instead of DeepSort
    https://www.pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/


    simple idea:
        a) for each frame always first update all old trackers with CSR-DCF https://arxiv.org/pdf/1611.08461.pdf
        b) for each tracker find detection with highest iou (at least 0.5)
        c) if tracker matched with detection:
              new general position is average of tracker position and matched detection
              replace tracker in multilist by init again with new averaged bounding box
              steps_without_detection = 0
              set active
        d) else:
              steps_without_detection++
              if steps_without_detection>thresh: set inactive
        e) reassign: for each unmatched detection, that doesn't has high iou with any track
            if len(active tracks)==upper_bound:
                calc center distance to all inactive tracks, merge with nearest track
            else:
                add new track if position not close to other track
"""

import argparse
import time
import cv2 as cv 
import numpy as np 
from glob import glob 
import os
from copy import deepcopy 
from collections import deque
from tqdm import tqdm

from lapsolver import solve_dense

from multitracker.tracking.deep_sort.application_util import preprocessing
from multitracker.tracking.deep_sort.application_util import visualization
from multitracker.tracking.deep_sort.deep_sort import nn_matching
from multitracker.tracking.deep_sort import deep_sort_app
from multitracker.tracking.deep_sort.deep_sort.detection import Detection

from multitracker import autoencoder
from multitracker.keypoint_detection import roi_segm, unet
from multitracker.tracking.inference import get_heatmaps_keypoints
from multitracker.tracking.keypoint_tracking import tracker as keypoint_tracking
from multitracker.be import video
from multitracker import util 
from multitracker.experiments.roi_curve import calc_iou
from multitracker.tracking import inference

colors = util.get_colors()



def tlbr2tlhw(tlbr):
    return [tlbr[0], tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1]]
def tlhw2tlbr(tlhw):
    return [tlhw[0], tlhw[1], tlhw[0]+tlhw[2], tlhw[1]+tlhw[3]]
def tlhw2chw(tlhw):
    return [ tlhw[0]+tlhw[2]/2. , tlhw[1]+tlhw[3]/2., tlhw[2], tlhw[3] ]

class OpenCVTrack(object):
    def __init__(self,track_id,tlhw,active,steps_without_detection,last_means,score):
        self.tlhw,self.active,self.steps_without_detection = tlhw,active,steps_without_detection
        self.track_id, self.time_since_update = track_id, steps_without_detection
        self.score = score
        self.last_means = last_means
        self.last_means = [ tlhw2chw(p) for p in self.last_means ]
        
    def is_confirmed(self):
        return self.active 
    def to_tlbr(self): 
        return tlhw2tlbr(self.tlhw)
    def to_tlwh(self):
        return self.tlhw

class Tracker(object):
    def __init__(self):
        self.tracks = [] 
        
    def associate(self, tracked_boxes, detected_boxes, sigma_iou = 0.25):
        """ perform association between tracks and detections in a frame.
        Args:
            tracks (list): input tracks
            detected_boxes (list): input detections
            sigma_iou (float): minimum intersection-over-union of a valid association
        Returns:
            (tuple): tuple containing:
            track_ids (numpy.array): 1D array with indexes of the tracks
            det_ids (numpy.array): 1D array of the associated indexes of the detections
        """
        costs = np.empty(shape=(len(tracked_boxes), len(detected_boxes)), dtype=np.float32)
        for row, track in enumerate(tracked_boxes):
            for col, dbox in enumerate(detected_boxes):
                iou = calc_iou(tlhw2tlbr(track),tlhw2tlbr(dbox))
                costs[row, col] = 1. - iou

        np.nan_to_num(costs)
        costs[costs > 1 - sigma_iou] = np.nan
        track_ids, det_ids = solve_dense(costs)
        return track_ids, det_ids
        
def get_cv_multitracker():
    try:
        return cv.MultiTracker_create()
    except:
        return cv.legacy.MultiTracker_create()

class UpperBoundTracker(Tracker):
    def __init__(self, upper_bound):
        super().__init__()
        self.upper_bound = upper_bound
        self.thresh_set_inactive = 12
        self.thresh_set_dead = 100
        self.maximum_other_track_init_distance = 150
        self.maximum_nearest_inactive_track_distance = 1000#500# self.maximum_other_track_init_distance * 4

        self.tracks = []
        self.trackers = get_cv_multitracker()
        self.active = [False for _ in range(upper_bound)]
        self.active_cnt = 0
        self.steps_without_detection = [0 for _ in range(upper_bound)]
        self.last_means = [[] for _ in range(upper_bound)]
        self.detection_buffer_length = 3
        self.detection_buffer = deque(maxlen=self.detection_buffer_length)
        print('[*] inited UpperBoundTracker with fixed number %s' % upper_bound)

    def step(self,ob):
        debug = bool(0) 
        frame = ob['img']
        [detected_boxes,scores, features] = ob['detections']
        self.detection_buffer.append(ob['detections'])

        # a) update all trackers of last frame
        (success, tracked_boxes) = self.trackers.update(frame)
        
        # b) then find for each tracker unmatched detection with highest iou (at least 0.5)
        matched_detections = [False for _ in detected_boxes]
        matched_tracks = [False for _ in range(self.upper_bound)]
        matched_track_scores = [False for _ in range(self.upper_bound)]
        
        ## linear assignment
        track_ids, det_ids = self.associate(tracked_boxes, detected_boxes)
        for i, (track_id, det_id) in enumerate(zip(track_ids, det_ids)):
            matched_detections[det_id] = True 
            matched_tracks[track_id] = True 

            # new general position is average of tracker prediction and matched detection
            alpha = 0.5
            box = alpha * tracked_boxes[track_id] + (1.-alpha) * detected_boxes[det_id]
            self.last_means[track_id].append(box)
            
            # replace tracker in multilist by init again with new general bounding box
            if debug: print('[*]   updated active tracker %i with detection %i by matching' % (track_id, det_id),'det:',detected_boxes[det_id],'track:',tracked_boxes[track_id],'new:',box)
            tracked_boxes[track_id] = box
            matched_track_scores[track_id] = scores[det_id]
            
            self.steps_without_detection[track_id] = 0
            if not self.active[track_id]:
                self.active[track_id] = True
                self.active_cnt += 1
        
        self.trackers = get_cv_multitracker()
        for box in tracked_boxes:
            try:
                self.trackers.add(cv.legacy.TrackerCSRT_create(), frame, tuple([int(cc) for cc in box]))
            except:
                self.trackers.add(cv.TrackerCSRT_create(), frame, tuple([int(cc) for cc in box]))
        
        for i in range(len(tracked_boxes)):
            if not matched_tracks[i]:
                self.last_means[i].append(tracked_boxes[i])
                self.steps_without_detection[i] += 1 
                if self.steps_without_detection[i] > self.thresh_set_inactive and self.active[i]:
                    self.active[i] = False
                    self.active_cnt -= 1

        # e) reassign: check if unmatched detection is a new track or gets assigned to inactive track (depending on barrier fixed number)
        #for j in range(min(len(detected_boxes),self.upper_bound)):
        for j in range(len(detected_boxes)):
            dbox = detected_boxes[j] # boxes are sorted desc as scores!
            if not matched_detections[j] and ob['frame_idx']>3: 
                ## try to filter out flickering false positive detections by only using detections that can be greedily iou matched 3 steps through time
                stable_detection = True 
                xd1, yd1, wd1, hd1 = dbox
                for step_back in range(1,1+min(ob['frame_idx'],self.detection_buffer_length)):
                    [last_detected_boxes, last_scores, last_features] = self.detection_buffer[-step_back]
                    _stable_detection = False
                    for jo in range(len(last_detected_boxes)):
                        xd2, yd2, wd2, hd2 = last_detected_boxes[jo]
                        if last_scores[jo] > 0.5:
                            iou = calc_iou([yd1,xd1,yd1+hd1,xd1+wd1], [yd2,xd2,yd2+hd2,xd2+wd2])
                            if iou>0.5:
                                _stable_detection = True 
                    stable_detection = stable_detection and _stable_detection
                if stable_detection:

                    if len(self.trackers.getObjects()) < self.upper_bound:
                        # check if appropiate minimum distance to other track before initiating
                        other_track_near = False 
                        for tbox in self.trackers.getObjects():
                            other_track_near = other_track_near or np.linalg.norm(dbox-tbox) < self.maximum_other_track_init_distance*frame.shape[0]/1080.
                                
                        if not other_track_near:
                            # add new track
                            dboxi = tuple([int(cc) for cc in dbox])
                            try:
                                self.trackers.add(cv.legacy.TrackerCSRT_create(), frame, dboxi)
                            except:
                                self.trackers.add(cv.TrackerCSRT_create(), frame, dboxi)
                            self.active[self.active_cnt] = True 
                            if debug: print('[*]   added tracker %i with detection %i' % (self.active_cnt,j))
                            self.active_cnt += 1
                    else:
                        # only consider reactivating old track with this detection if detected box not high iou with any track
                        other_track_overlaps = False 
                        for k, tbox in enumerate(self.trackers.getObjects()):
                            other_track_overlaps = other_track_overlaps or calc_iou(tlhw2tlbr(dbox),tlhw2tlbr(tbox)) > 0.15 
                        if not other_track_overlaps:
                            # calc center distance to all inactive tracks, merge with nearest track
                            nearest_inactive_track_distance, nearest_inactive_track_idx = 1e7,-1
                            for i, tbox in enumerate(self.trackers.getObjects()):
                                if not self.active[i]:
                                    (detx, dety, detw, deth) = dbox 
                                    (trackx,tracky,trackw,trackh) = tbox
                                    dist = np.sqrt(((detx+detw/2.)-(trackx+trackw/2.))**2 + ((dety+deth/2.) -(tracky+trackh/2.))**2 )
                                    if nearest_inactive_track_distance > dist:
                                        nearest_inactive_track_distance, nearest_inactive_track_idx = dist, i
                            
                            # merge by initing tracker with this detected box
                            # replace tracker in multilist by init again with new general bounding box
                            obs = self.trackers.getObjects()
                            #if nearest_inactive_track_idx >= 0:
                            if nearest_inactive_track_distance < self.maximum_nearest_inactive_track_distance * frame.shape[0]/1000.:
                                ## the old estimation was obvisiouly wrong, so correct last means and add interpolated track
                                try:
                                    self.last_means[nearest_inactive_track_idx] = self.last_means[nearest_inactive_track_idx][:-self.steps_without_detection[nearest_inactive_track_idx]]
                                    for ims in range(self.steps_without_detection[nearest_inactive_track_idx]):
                                        ratio = ims / float(self.steps_without_detection[nearest_inactive_track_idx])
                                        interpolated_box = (1. - ratio) * self.last_means[nearest_inactive_track_idx][-1] + ratio * detected_boxes[j]
                                        self.last_means[nearest_inactive_track_idx].append(interpolated_box)
                                except Exception as e:
                                    print('[* ERROR %s] when correcting wrong path'% ob['frame_idx'],e)
                                matched_detections[j] = True
                                matched_tracks[nearest_inactive_track_idx] = True
                                obs[nearest_inactive_track_idx] = detected_boxes[j]
                                self.active[nearest_inactive_track_idx] = True
                                self.active_cnt += 1 
                                self.steps_without_detection[nearest_inactive_track_idx] = 0
                                self.trackers = get_cv_multitracker()
                                for _ob in obs:
                                    try:
                                        self.trackers.add(cv.legacy.TrackerCSRT_create(), frame, tuple([int(cc) for cc in _ob]))
                                    except:
                                        self.trackers.add(cv.TrackerCSRT_create(), frame, tuple([int(cc) for cc in _ob]))


                                if debug: print('[*]   updated inactive tracker %i with detection %i' % (nearest_inactive_track_idx,j))

        # update internal variables to be compatible with rest
        self.tracks = []
        for i, tbox in enumerate(self.trackers.getObjects()):
            self.tracks.append(OpenCVTrack(i,tbox,self.active[i],self.steps_without_detection[i],self.last_means[i],matched_track_scores[i]))

        if debug:
            for i, tbox in enumerate(self.trackers.getObjects()):
                if len(self.last_means[i])>0:
                    print('Tracker',i,self.last_means[i][-1],'active',self.active[i],'steps misses',self.steps_without_detection[i])
            for j, dbox in enumerate(detected_boxes):
                print('Detect',j,str(scores[j])[1:4],dbox)
            print()


def run(config, detection_model, encoder_model, keypoint_model, min_confidence_boxes, min_confidence_keypoints, tracker = None):
    assert 'upper_bound' in config and config['upper_bound'] is not None and int(config['upper_bound'])>0
    #config['upper_bound'] = None # ---> force VIOU tracker
    
    nms_max_overlap = 1.
    nms_max_overlap = .25

    video_reader = cv.VideoCapture( config['video'] )
    # ignore first 5 frames
    for _ in range(5):
        ret, frame = video_reader.read()
    [Hframe,Wframe,_] = frame.shape
    crop_dim = roi_segm.get_roi_crop_dim(config['data_dir'], config['project_id'], config['test_video_ids'].split(',')[0],Hframe)
    total_frame_number = int(video_reader.get(cv.CAP_PROP_FRAME_COUNT))
    print('[*] total_frame_number',total_frame_number,'Hframe,Wframe',Hframe,Wframe,'crop_dim',crop_dim)
    
    video_file_out = inference.get_video_output_filepath(config)
    if config['file_tracking_results'] is None:
        config['file_tracking_results'] = video_file_out.replace('.%s'%video_file_out.split('.')[-1],'.csv')
    file_csv = open( config['file_tracking_results'], 'w') 
    file_csv.write('video_id,frame_id,track_id,center_x,center_y,x1,y1,x2,y2,time_since_update\n')
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
    tracker = UpperBoundTracker(config['upper_bound'])
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

    #while running: #video_reader.isOpened():
    for frame_idx in tqdm(range(total_frame_number)):
        #frame_idx += 1 
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
                file_csv.close()
                return True  
        else:
            running = False 
            file_csv.close()
            return True 
        
        showing = True # frame_idx % 1000 == 0

        if running:
            if len(detection_buffer) == 0:
                frames_tensor = np.array(list(frame_buffer)).astype(np.float32)
                # fill up frame buffer and then detect boxes for complete frame buffer
                t_odet_inf_start = time.time()
                batch_detections = inference.detect_batch_bounding_boxes(config, detection_model, frames_tensor, min_confidence_boxes)
                [detection_buffer.append(batch_detections[ib]) for ib in range(config['inference_objectdetection_batchsize'])]
                t_odet_inf_end = time.time()
                if frame_idx < 200 and frame_idx % 10 == 0:
                    print('  object detection ms',(t_odet_inf_end-t_odet_inf_start)*1000.,"batch", len(batch_detections),len(detection_buffer), (t_odet_inf_end-t_odet_inf_start)*1000./len(batch_detections) ) #   roughly 70ms

                if keypoint_model is not None:
                    t_kp_inf_start = time.time()
                    keypoint_buffer = inference.inference_batch_keypoints(config, keypoint_model, crop_dim, frames_tensor, detection_buffer, min_confidence_keypoints)
                    #[keypoint_buffer.append(batch_keypoints[ib]) for ib in range(config['inference_objectdetection_batchsize'])]
                    t_kp_inf_end = time.time()
                    if frame_idx < 200 and frame_idx % 10 == 0:
                        print('  keypoint ms',(t_kp_inf_end-t_kp_inf_start)*1000.,"batch", len(keypoint_buffer),(t_kp_inf_end-t_kp_inf_start)*1000./ (1e-6+len(keypoint_buffer)) ) #   roughly 70ms
                
            # if detection buffer not empty use preloaded frames and preloaded detections
            frame = frame_buffer.popleft()
            detections = detection_buffer.popleft()
                
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            features = np.array([d.feature for d in detections])
            # Update tracker
            tracker.step({'img':frame,'detections':[boxes, scores, features], 'frame_idx': frame_idx})

            if keypoint_model is not None:
                keypoints = keypoint_buffer.popleft()
                # update tracked keypoints with new detections
                tracked_keypoints = keypoint_tracker.update(keypoints)
            else:
                keypoints = tracked_keypoints = []

            # Store results.        
            for track in tracker.tracks:
                bbox = track.to_tlwh()
                center0, center1, _, _ = tlhw2chw(bbox)
                result = [video_id, frame_idx, track.track_id, center0, center1, bbox[0], bbox[1], bbox[2], bbox[3], track.time_since_update]
                file_csv.write(','.join([str(r) for r in result])+'\n')
                results.append(result)
            
            #print('[%i/%i] - %i detections. %i keypoints' % (config['count'], total_frame_number, len(detections), len(keypoints)))
            if showing:
                out = deep_sort_app.visualize(visualizer, frame, tracker, detections, keypoint_tracker, keypoints, tracked_keypoints, crop_dim, results, sketch_file=config['sketch_file'])
                video_writer.writeFrame(cv.cvtColor(out, cv.COLOR_BGR2RGB))
            
            if int(frame_idx) == 1010:
                tbenchend = time.time()
                print('[*] 1000 steps took',tbenchend-tbenchstart,'seconds')
                step_dur_ms = 1000.*(tbenchend-tbenchstart)/1000.
                fps = 1. / ( (tbenchend-tbenchstart)/1000. )
                print('[*] one time step takes on average',step_dur_ms,'ms',fps,'fps')

            if showing:
                cv.imshow("tracking visualization", out)#cv.resize(out,None,None,fx=0.75,fy=0.75))
                cv.waitKey(1)
        