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
    
    def __init__(self,track_id,tlhw,active=True,score=0.5,history_length=50):
        self.tlhw,self.active = tlhw,active
        self.track_id = track_id
        self.score = score
        
        self.steps_age = 0 
        self.steps_unmatched = 0
        self.active = True 
        self.history = deque(maxlen=history_length)

    def update(self, tlhw, global_step):
        self.tlhw = tlhw 
        self.steps_age += 1 
        self.steps_unmatched = 0 
        self.active = True 
        self.history.append({'global_step': global_step, 'bbox': tlhw, 'matched': True})
    
    def mark_missed(self, global_step):
        # update position to continue linear movement
        if len(self.history) > 0:
            self.tlhw += self.tlhw - self.history[-1]['bbox']
        self.steps_age += 1 
        self.steps_unmatched += 1 
        self.history.append({'global_step': global_step, 'bbox': self.tlhw, 'matched': False})

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
        tc0 = time.time()
        costs = np.empty(shape=(len(tracked_boxes), len(detected_boxes)), dtype=np.float32)
        for row, track in enumerate(tracked_boxes):
            tbox = tlhw2tlbr(track.tlhw)
            for col, dbox in enumerate(detected_boxes):
                iou = calc_iou(tbox,tlhw2tlbr(dbox))
                costs[row, col] = 1. - iou
        np.nan_to_num(costs)
        costs[costs > 1 - sigma_iou] = np.nan
        tc1 = time.time()
        th0 = time.time()
        track_ids, det_ids = solve_dense(costs)
        th1 = time.time()
        #print('matching cost',1000.*(tc1-tc0),'optim',1000.*(th1-th0))
        return track_ids, det_ids

class UpperBoundTracker(Tracker):
    def __init__(self, upper_bound, config = None):
        super().__init__()
        self.upper_bound = upper_bound
        default_config = {
            "thresh_set_inactive": 12, # after how many time steps of no matching do we consider a track lost/inactive
            "thresh_set_dead": 100, # after how many time steps of no matching do we delete a track
            "maximum_other_track_init_distance": 0.1, # maximum distance (relative to image size) between a track and a detection for new track creation
            "maximum_nearest_reassign_track_distance": 0.9, # maximum distance (relative to image size) between a track and a detection for reassignment
            "track_history_length": 1000, # ideally longer than maximum occlusion time
            "matching_iou_thresh": 0.1, # min IoU to match old track with new detection
            "stabledetection_bufferlength": 25, # how many time steps do we check for a stable detection before creating new track
            "stabledetection_iou_thresh": 0.2, # min IoU between detections to be considered stable enough for track creation
            "stabledetection_confidence_thresh": 0.5 # min detection confidence to be considered for a stable detection
        }
        self.config = [config, default_config][int(config is None)] 
        self.detection_buffer = deque(maxlen=self.config['stabledetection_bufferlength'])
        self.tracks = []
        self.track_count = 0 
        self.global_step = -1
    
    def step(self,ob):
        self.global_step += 1
        debug = bool(1) 
        frame = ob['img']
        self.frame_shape = frame.shape
        [detected_boxes,scores, features] = ob['detections']
        self.detection_buffer.append(ob['detections'])

        matched_detections, matched_tracks = {},{}

        ## matching from old tracks with new detections
        if min(len(self.tracks),len(detected_boxes)) > 0:
            track_ids, det_ids = self.associate(self.tracks, detected_boxes, self.config['matching_iou_thresh'])
            for i, (track_idx, det_idx) in enumerate(zip(track_ids, det_ids)):
                matched_detections[det_idx] = True 
                matched_tracks[track_idx] = True 
                self.tracks[track_idx].update(detected_boxes[det_idx], self.global_step)
                #if debug: print('[*] matched det',detected_boxes[det_idx],'to track',track_idx)

        for i in range(len(self.tracks)):
            if i not in matched_tracks:
                self.tracks[i].mark_missed(self.global_step)
                matched_tracks[i] = False 
            
                if self.tracks[i].steps_unmatched > self.config['thresh_set_inactive']:
                    self.tracks[i].active = False
                
                #if debug: print('[*] unmatched track',i,self.tracks[i].steps_unmatched,self.tracks[i].active)

        # check if unmatched detection is a new track or gets assigned to inactive track (depending on upper bound number)
        for j in range(len(detected_boxes)):
            dbox = detected_boxes[j] # boxes are sorted desc as scores!
            if j not in matched_detections: matched_detections[j] = False 

            if not matched_detections[j] and self.global_step > self.config['stabledetection_bufferlength']: 
                ## stable detection check
                ##   try to filter out flickering false positive detections by only using detections that can be greedily iou 
                ##   matched some steps through time
                stable_detection = True 
                xd1, yd1, wd1, hd1 = dbox
                stable_path = []
                for step_back in range(1,self.config['stabledetection_bufferlength']):
                    [last_detected_boxes, last_scores, last_features] = self.detection_buffer[-step_back]
                    _stable_detection = False
                    for jo in range(len(last_detected_boxes)):
                        xd2, yd2, wd2, hd2 = last_detected_boxes[jo]
                        if not _stable_detection and last_scores[jo] > self.config['stabledetection_confidence_thresh']:
                            iou = calc_iou([yd1,xd1,yd1+hd1,xd1+wd1], [yd2,xd2,yd2+hd2,xd2+wd2])
                            if iou > self.config['stabledetection_iou_thresh']:
                                _stable_detection = True 
                                stable_path.insert(0,[xd2, yd2, wd2, hd2])
                                xd1, yd1, wd1, hd1 = xd2, yd2, wd2, hd2

                    stable_detection = stable_detection and _stable_detection
                if stable_detection:
                    ## if upper bound of tracks not yet reached, we are allowed to add this stable detection as a new track
                    if len(self.tracks) < self.upper_bound:
                        # check if appropiate minimum distance to other track before initiating
                        other_track_near = False 
                        for tbox in self.tracks:
                            other_track_near = other_track_near or np.linalg.norm(dbox[:2]-tbox.tlhw[:2]) < self.config['maximum_other_track_init_distance']*min(self.frame_shape[:2])
                                
                        if not other_track_near:
                            # add new track
                            self.tracks.append(OpenCVTrack(self.track_count, dbox, history_length = self.config['track_history_length']))
                            self.track_count += 1 
                                
                            #self.active[self.active_cnt] = True 
                            if debug: print('[*]   added tracker %i with detection %i' % (self.tracks[-1].track_id,j))
                            #self.active_cnt += 1
                    else:
                        ## Upper Bound Violation: despite all tracks already existent, there is another stable detection
                        # only consider reassigning old track with this detection if detected box not high iou with any track (might be cluttered detections)
                        other_track_overlaps = False 
                        for k, _track in enumerate(self.tracks):
                            other_track_overlaps = other_track_overlaps or calc_iou(tlhw2tlbr(dbox),tlhw2tlbr(_track.tlhw)) > self.config['stabledetection_iou_thresh']
                        if not other_track_overlaps:
                            # calc center distance to all inactive tracks, merge with nearest track if close enough
                            det_center = np.array([dbox[0]+dbox[2]/2.,dbox[1]+dbox[3]/2.])
                            closest_possible_tracks = [
                                {'idx': kk, 'track': t, 'dist': np.linalg.norm( np.array([t.tlhw[1]+t.tlhw[3]/2.,t.tlhw[0]+t.tlhw[2]/2.])-det_center ) } 
                                for kk,t in enumerate(self.tracks) if t.active == False ]
                            closest_possible_tracks = [d for d in closest_possible_tracks if d['dist'] < self.config['maximum_nearest_reassign_track_distance'] * min(self.frame_shape[:2])]# or self.global_step < 100]
                            closest_possible_tracks = sorted(closest_possible_tracks, key = lambda d: d['dist'])
                            
                            if len(closest_possible_tracks) > 0:
                                ## reassign stable detection to closest possible track:
                                ##   - delete wrong history of track (since the track was matched the last time)
                                ##   - add linear interpolation between last bbox track was matched with beginning of stable detection history
                                ##   - add stable detection history

                                # delete wrong history of track (since the track was matched the last time)
                                num_delete_history = 0
                                while not closest_possible_tracks[0]['track'].history[-1]['matched']: 
                                    closest_possible_tracks[0]['track'].history.pop()
                                    num_delete_history += 1 

                                # add linear interpolation between last bbox track was matched with beginning of stable detection history
                                step_start_inter = self.global_step - num_delete_history
                                step_end_inter = self.global_step - self.config['stabledetection_bufferlength']
                                start_box = np.array(closest_possible_tracks[0]['track'].history[-1]['bbox'])
                                end_box = np.array(stable_path[0])
                                for inter_step in range(step_start_inter, step_end_inter):
                                    ratio = (inter_step-step_start_inter) / (step_end_inter-step_start_inter)
                                    inter_tlhw = start_box + ratio * (end_box - start_box)
                                    closest_possible_tracks[0]['track'].history.append({'global_step': inter_step, 'bbox': inter_tlhw, 'matched': False})
    
                                # add stable detection history
                                for istable in range(min(len(stable_path),self.config['stabledetection_bufferlength'])):
                                    closest_possible_tracks[0]['track'].history.append({'global_step': step_end_inter+istable, 'bbox': stable_path[istable], 'matched': True})

                                ## update closest possibly assigned track with current position
                                matched_detections[j] = True
                                matched_tracks[closest_possible_tracks[0]['idx']] = True
                                self.tracks[closest_possible_tracks[0]['idx']].update(detected_boxes[j], self.global_step)
                                self.tracks[closest_possible_tracks[0]['idx']].steps_age += self.config['stabledetection_bufferlength'] - num_delete_history

                                if debug: print('[*]   reassigned inactive tracker %i with detection %i' % (closest_possible_tracks[0]['idx'],j))

        
def run(config, detection_model, encoder_model, keypoint_model, min_confidence_boxes, min_confidence_keypoints, tracker = None):
    assert 'upper_bound' in config and config['upper_bound'] is not None and int(config['upper_bound'])>0, "ERROR: Upper Bound Tracking requires the argument --upper_bound to bet set (eg --upper_bound 4)"
    #config['upper_bound'] = None # ---> force VIOU tracker
    
    nms_max_overlap = 1.
    nms_max_overlap = .25

    video_reader = cv.VideoCapture( config['video'] )
    # ignore first 5 frames
    #for _ in range(5):
    #    ret, frame = video_reader.read()
    
    Wframe  = int(video_reader.get(cv.CAP_PROP_FRAME_WIDTH))
    Hframe = int(video_reader.get(cv.CAP_PROP_FRAME_HEIGHT))
    
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
                return True  
        else:
            running = False 
            file_csv.close()
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
            tracker.step({'img':frame,'detections':[boxes, scores, features], 'frame_idx': frame_idx})
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
                center0, center1, _, _ = tlhw2chw(bbox)
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
                cv.imshow("tracking visualization", out)#cv.resize(out,None,None,fx=0.75,fy=0.75))
                cv.waitKey(1)
        
            if 0:
                dur_imread = timread1 - timread0
                dur_obdet = tobdet1 - tobdet0
                dur_obtrack = tobtrack1 - tobtrack0
                dur_kptrack = tkptrack1 - tkptrack0
                dur_vis = tvis1 - tvis0
                dur_imread, dur_obdet, dur_obtrack, dur_kptrack, dur_vis = [1000. * dur for dur in [dur_imread, dur_obdet, dur_obtrack, dur_kptrack, dur_vis]]
                print('imread',dur_imread,'obdetect',dur_obdet, 'obtrack',dur_obtrack, 'kptrack',dur_kptrack, 'vis',dur_vis)