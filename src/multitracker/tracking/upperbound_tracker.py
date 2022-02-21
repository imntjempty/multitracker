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
import pandas as pd

from multitracker.tracking.deep_sort.deep_sort.detection import Detection
from multitracker import util 
from multitracker.experiments.roi_curve import calc_iou

colors = util.get_colors()

should_correct_csv = True


class Track(object):
    
    def __init__(self,track_id,tlhw,active=True,score=0.5,history_length=50):
        self.tlhw,self.active = tlhw,active
        self.track_id = track_id
        self.score = score
        
        self.steps_age = 0 
        self.steps_unmatched = 0
        self.speed_px = 0 # speed in pixel/sec
        self.active = True 
        self.totaltraveled_px = 0 
        self.history = deque(maxlen=history_length)

    def update(self, tlhw, global_step):
        self.tlhw = tlhw 
        self.steps_age += 1 
        self.steps_unmatched = 0 
        self.active = True 
        if len(self.history) > 0:
            deltabox = self.tlhw - self.history[-1]['bbox']
            self.speed_px = np.mean(np.abs(deltabox[:2]+deltabox[2:4]/2.))
            self.totaltraveled_px += self.speed_px
        self.history.append({'global_step': global_step, 'bbox': tlhw, 'matched': True, 'speed_px': self.speed_px})
    
    def mark_missed(self, global_step):
        # update position to continue linear movement (continue with same speed)
        if len(self.history) > 0:
            deltabox = self.tlhw - self.history[-1]['bbox']
            self.tlhw += deltabox
            self.totaltraveled_px += self.speed_px
        self.steps_age += 1 
        self.steps_unmatched += 1 
        self.history.append({'global_step': global_step, 'bbox': self.tlhw, 'matched': False, 'speed_px': self.speed_px})

    def is_confirmed(self):
        return self.active 
    def to_tlbr(self): 
        return util.tlhw2tlbr(self.tlhw)
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
            tbox = util.tlhw2tlbr(track.tlhw)
            for col, dbox in enumerate(detected_boxes):
                iou = calc_iou(tbox, util.tlhw2tlbr(dbox))
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
    def __init__(self, config):
        super().__init__()
        self.upper_bound = config['upper_bound']
        default_config = {
            "thresh_set_inactive": 12, # after how many time steps of no matching do we consider a track lost/inactive
            "maximum_nearest_reassign_track_distance": 2, # maximum distance (relative to image size) between a track and a detection for reassignment
            "thresh_set_dead": 100, # after how many time steps of no matching do we delete a track
            "track_history_length": 1000, # ideally longer than maximum occlusion time
            "matching_iou_thresh": 0.1, # min IoU to match old track with new detection
            "maximum_other_track_init_distance": 1.0, # maximum distance (relative to image size) between a track and a detection for new track creation
            "stabledetection_bufferlength": 10, # how many time steps do we check for a stable detection before creating new track
            "stabledetection_iou_thresh": 0.2, # min IoU between detections to be considered stable enough for track creation
            "stabledetection_confidence_thresh": 0.3 # min detection confidence to be considered for a stable detection
        }
        self.config = config #[config, default_config][int(config is None)] 
        for k in default_config.keys():
            if k not in self.config:
                self.config[k] = default_config[k]

        self.detection_buffer = deque(maxlen=self.config['stabledetection_bufferlength'])
        self.tracks = []
        self.track_count = 0 
        self.global_step = -1
    
    def step(self,ob):
        self.global_step += 1
        debug = bool(0) 
        frame = ob['img']
        self.frame_shape = frame.shape
        [detections, detected_boxes,scores, features] = ob['detections']
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
                    [detections, last_detected_boxes, last_scores, last_features] = self.detection_buffer[-step_back]
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
                            other_track_near = other_track_near or np.linalg.norm(dbox[:2]-tbox.tlhw[:2]) > self.config['maximum_other_track_init_distance']*min(self.frame_shape[:2])
                                
                        if not other_track_near:
                            # add new track
                            self.tracks.append(Track(self.track_count, dbox, history_length = self.config['track_history_length']))
                            self.track_count += 1 
                                
                            #self.active[self.active_cnt] = True 
                            if debug: print('[*]   added tracker %i with detection %i' % (self.tracks[-1].track_id,j))
                            #self.active_cnt += 1
                    else:
                        ## Upper Bound Violation: despite all tracks already existent, there is another stable detection
                        # only consider reassigning old track with this detection if detected box not high iou with any track (might be cluttered detections)
                        other_track_overlaps = False 
                        for k, _track in enumerate(self.tracks):
                            other_track_overlaps = other_track_overlaps or calc_iou(util.tlhw2tlbr(dbox),util.tlhw2tlbr(_track.tlhw)) > self.config['stabledetection_iou_thresh']
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
                                if hasattr(closest_possible_tracks[0]['track'],'history') and len(closest_possible_tracks[0]['track'].history) > 0:
                                    while len(closest_possible_tracks[0]['track'].history) > 0 and not closest_possible_tracks[0]['track'].history[-1]['matched']: 
                                        popped = closest_possible_tracks[0]['track'].history.pop()
                                        num_delete_history += 1 

                                if should_correct_csv:
                                    # delete num_delete_history lines in csv with closest_possible_track id
                                    with open(ob['file_tracking_results'],'r') as fread:
                                        csv_dat = fread.readlines() #[ l.replace('\n','') for l in fread.readlines()]
                                        header_elems = csv_dat[0].split(',') # video_id,frame_id,track_id,center_x,center_y,x1,y1,x2,y2,time_since_update
                                        _cntcsv = 0 
                                        last_frame_id = csv_dat[-1].split(',')[1]
                                        line_pointer = len(csv_dat)-2
                                        inlen = len(csv_dat)-1
                                        done = False
                                        while not done:
                                            video_id,frame_id,track_id,center_x,center_y,x1,y1,w,h,time_since_update =  csv_dat[line_pointer].split(',')
                                            # if match in time and track id, delete
                                            if int(frame_id) > int(last_frame_id)-num_delete_history  and int(track_id) == int(closest_possible_tracks[0]['track'].track_id):
                                                #print('[*] deleting csv line', line_pointer, csv_dat[line_pointer][:-1])
                                                del csv_dat[line_pointer]
                                            if int(frame_id) <= int(last_frame_id) - num_delete_history or line_pointer == 1:
                                                done = True 
                                            line_pointer -= 1
                                    
                                # save updated csvs back to disk
                                with open(ob['file_tracking_results'],'w') as fwrite:
                                    for l in csv_dat:
                                        fwrite.write(l)
                                
                                cnt_addedlines = 0
                                track_id = closest_possible_tracks[0]['track'].track_id
                                # add linear interpolation between last bbox track was matched with beginning of stable detection history
                                step_start_inter = self.global_step - num_delete_history
                                step_end_inter = self.global_step - self.config['stabledetection_bufferlength']
                                if hasattr(closest_possible_tracks[0]['track'],'history') and len(closest_possible_tracks[0]['track'].history) > 0:
                                    start_box = np.array(closest_possible_tracks[0]['track'].history[-1]['bbox'])
                                else:
                                    start_box = np.array(closest_possible_tracks[0]['track'].tlhw)
                                end_box = np.array(stable_path[0])
                                
                                for inter_step in range(step_start_inter, step_end_inter):
                                    ratio = (inter_step-step_start_inter) / (step_end_inter-step_start_inter)
                                    inter_tlhw = start_box + ratio * (end_box - start_box)
                                    closest_possible_tracks[0]['track'].history.append({'global_step': inter_step, 'bbox': inter_tlhw, 
                                        'matched': False, 'speed_px': closest_possible_tracks[0]['track'].speed_px})
                                    
                                    if should_correct_csv:
                                        with open(ob['file_tracking_results'],'a') as fappend:
                                            y1, x1, h, w = inter_tlhw
                                            y2, x2 = y1+h, x1+w
                                            time_since_update = closest_possible_tracks[0]['track'].steps_unmatched
                                            fappend.write(','.join([str(q) for q in [-1,inter_step,track_id,x1 + (x2-x1)/2.,y1 + (y2-y1)/2.,x1,y1,w,h,time_since_update]])+'\n')
                                            cnt_addedlines += 1 
                                
                                # add stable detection history
                                for istable in range(min(len(stable_path),self.config['stabledetection_bufferlength'])):
                                    closest_possible_tracks[0]['track'].history.append({'global_step': step_end_inter+istable, 'bbox': stable_path[istable], 
                                        'matched': True, 'speed_px': closest_possible_tracks[0]['track'].speed_px})

                                    if should_correct_csv:
                                        with open(ob['file_tracking_results'],'a') as fappend:
                                            track_id = closest_possible_tracks[0]['track'].track_id
                                            y1,x1,h,w = stable_path[istable]
                                            y2, x2 = y1+h,x1+w
                                            time_since_update = closest_possible_tracks[0]['track'].steps_unmatched
                                            fappend.write(','.join([str(q) for q in [-1,step_end_inter+istable,track_id,x1 + (x2-x1)/2.,y1 + (y2-y1)/2.,x1,y1,w,h,time_since_update]])+'\n')
                                            cnt_addedlines += 1
                                
                                ## sort csv lines by frame_id 
                                with open(ob['file_tracking_results'],'r') as fread:
                                    csv_data = fread.readlines()
                                csv_data = [csv_data[0]] + sorted(csv_data[1:], key = lambda line: line.split(',')[1])
                                os.remove(ob['file_tracking_results'])
                                with open(ob['file_tracking_results'],'w') as fo:
                                    for line in csv_data:
                                        fo.write(line)

                                ## update closest possibly assigned track with current position
                                matched_detections[j] = True
                                matched_tracks[closest_possible_tracks[0]['idx']] = True
                               
                                self.tracks[closest_possible_tracks[0]['idx']].update(detected_boxes[j], self.global_step)
                                self.tracks[closest_possible_tracks[0]['idx']].steps_age += self.config['stabledetection_bufferlength'] - num_delete_history

                                if debug: print('[*]   reassigned inactive tracker %i with detection %i' % (closest_possible_tracks[0]['idx'],j))

      