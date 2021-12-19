""" 
    
    Visual intersection-over-union tracker
    source https://github.com/bochinski/iou-tracker/blob/master/viou_tracker.py 

    IoUTracker http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf 
    VIoUTracker http://elvera.nue.tu-berlin.de/files/1547Bochinski2018.pdf 
"""

import os
from collections import deque
import cv2 as cv 
import numpy as np
from lapsolver import solve_dense
from tqdm import tqdm
from time import time
from glob import glob 

from multitracker.tracking.upperbound_tracker import Tracker, Track
from multitracker.util import tlhw2chw, iou
from multitracker.tracking.viou.vis_tracker import VisTracker

class VIoUTracker(Tracker):
    def __init__(self, config):
        """ V-IOU Tracker.
        See "Extending IOU Based Multi-Object Tracking by Visual Information by E. Bochinski, T. Senst, T. Sikora" for
        more information.
        Args:
            detections (list): list of detections per frame, usually generated by util.load_mot
            sigma_l (float): low detection threshold.
            sigma_h (float): high detection threshold.
            sigma_iou (float): IOU threshold.
            t_min (float): minimum track length in frames.
            ttl (float): maximum number of frames to perform visual tracking.
                        this can fill 'gaps' of up to 2*ttl frames (ttl times forward and backward).
            tracker_type (str): name of the visual tracker to use. see VisTracker for more details.
            keep_upper_height_ratio (float): float between 0.0 and 1.0 that determines the ratio of height of the object
                                            to track to the total height of the object used for visual tracking.
        Returns:
            list: list of tracks.
        """
        default_params = {
            'sigma_l': 0,
            'sigma_h': 0.5, 
            'sigma_iou': 0.5, 
            't_min': 2, 
            'ttl': 1, 
            'tracker_type': 'CSRT', 
            'keep_upper_height_ratio': 1 
        }
        for k in default_params.keys():
            if k not in config:
                config[k] = default_params[k]
        self.config = config 
        self.sigma_l, self.sigma_h, self.sigma_iou, self.t_min, self.ttl, self.tracker_type, self.keep_upper_height_ratio = config['sigma_l'], config['sigma_h'], config['sigma_iou'], config['t_min'], config['ttl'], config['tracker_type'], config['keep_upper_height_ratio']
        if self.tracker_type == 'NONE':
            assert self.ttl == 1, "ttl should not be larger than 1 if no visual tracker is selected"

        self.tracks_active = []
        self.tracks_extendable = []
        self.tracks_finished = []
        self.frame_buffer = []
        self.last_means = {}
        self.total_count = 0 
        
    def step(self, ob):
        debug = bool(0) 
        frame = ob['img']
        [detections, detected_boxes,scores, features] = ob['detections']
        frame_num = ob['frame_idx']

        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.ttl + 1:
            self.frame_buffer.pop(0)

        # apply low threshold to detections
        #dets = [det for det in detections_frame if det['score'] >= sigma_l]
        dets = []
        for i in range(len(scores)):
            if scores[i] > self.sigma_l:
                dets.append({
                    'bbox': tuple(detected_boxes[i]),
                    'score': scores[i],
                    'class': 1 # we only track one class of animals
                })

        track_ids, det_ids = self.associate([Track(-1,t['bboxes'][-1]) for t in self.tracks_active], [d['bbox'] for d in dets], self.sigma_iou)
        updated_tracks = []
        for track_id, det_id in zip(track_ids, det_ids):
            self.tracks_active[track_id]['bboxes'].append(dets[det_id]['bbox'])
            self.tracks_active[track_id]['max_score'] = max(self.tracks_active[track_id]['max_score'], dets[det_id]['score'])
            self.tracks_active[track_id]['classes'].append(dets[det_id]['class'])
            self.tracks_active[track_id]['det_counter'] += 1

            if self.tracks_active[track_id]['ttl'] != self.ttl:
                # reset visual tracker if active
                self.tracks_active[track_id]['ttl'] = self.ttl
                self.tracks_active[track_id]['visual_tracker'] = None

            updated_tracks.append(self.tracks_active[track_id])

        tracks_not_updated = [self.tracks_active[idx] for idx in set(range(len(self.tracks_active))).difference(set(track_ids))]

        for track in tracks_not_updated:
            if track['ttl'] > 0:
                if track['ttl'] == self.ttl:
                    # init visual tracker
                    track['visual_tracker'] = VisTracker(self.tracker_type, track['bboxes'][-1], self.frame_buffer[-2], self.keep_upper_height_ratio)
                # viou forward update
                ok, bbox = track['visual_tracker'].update(frame)

                if not ok:
                    # visual update failed, track can still be extended
                    self.tracks_extendable.append(track)
                    continue

                track['ttl'] -= 1
                track['bboxes'].append(bbox)
                updated_tracks.append(track)
            else:
                self.tracks_extendable.append(track)

        # update the list of extendable tracks. tracks that are too old are moved to the finished_tracks. this should
        # not be necessary but may improve the performance for large numbers of tracks (eg. for mot19)
        tracks_extendable_updated = []
        for track in self.tracks_extendable:
            if track['start_frame'] + len(track['bboxes']) + self.ttl - track['ttl'] >= frame_num:
                tracks_extendable_updated.append(track)
            elif track['max_score'] >= self.sigma_h and track['det_counter'] >= self.t_min:
                self.tracks_finished.append(track)
            #elif track['det_counter'] < self.t_min:
            #    self.total_count -= 1
        self.tracks_extendable = tracks_extendable_updated

        new_dets = [dets[idx] for idx in set(range(len(dets))).difference(set(det_ids))]
        dets_for_new = []

        for det in new_dets:
            finished = False
            # go backwards and track visually
            boxes = []
            vis_tracker = VisTracker(self.tracker_type, det['bbox'], frame, self.keep_upper_height_ratio)

            for f in reversed(self.frame_buffer[:-1]):
                ok, bbox = vis_tracker.update(f)
                if not ok:
                    # can not go further back as the visual tracker failed
                    break
                boxes.append(bbox)

                # sorting is not really necessary but helps to avoid different behaviour for different orderings
                # preferring longer tracks for extension seems intuitive, LAP solving might be better
                for track in sorted(self.tracks_extendable, key=lambda x: len(x['bboxes']), reverse=True):

                    offset = track['start_frame'] + len(track['bboxes']) + len(boxes) - frame_num
                    # association not optimal (LAP solving might be better)
                    # association is performed at the same frame, not adjacent ones
                    if 1 <= offset <= self.ttl - track['ttl'] and iou(track['bboxes'][-offset], bbox) >= self.sigma_iou:
                        if offset > 1:
                            # remove existing visually tracked boxes behind the matching frame
                            track['bboxes'] = track['bboxes'][:-offset+1]
                        track['bboxes'] += list(reversed(boxes))[1:]
                        track['bboxes'].append(det['bbox'])
                        track['max_score'] = max(track['max_score'], det['score'])
                        track['last_score'] = det['score']
                        track['classes'].append(det['class'])
                        track['ttl'] = self.ttl
                        track['visual_tracker'] = None

                        self.tracks_extendable.remove(track)
                        if track in self.tracks_finished:
                            del self.tracks_finished[self.tracks_finished.index(track)]
                        updated_tracks.append(track)

                        finished = True
                        break
                if finished:
                    break
            if not finished:
                dets_for_new.append(det)

        # create new tracks
        new_tracks = []
        for det in dets_for_new:
            new_tracks.append({'bboxes': [det['bbox']], 'max_score': det['score'], 'last_score':det['score'], 'start_frame': frame_num, 'ttl': self.ttl,
                    'classes': [det['class']], 'det_counter': 1, 'visual_tracker': None, 'track_id': self.total_count})
            self.total_count += 1 
        self.tracks_active = []
        for track in updated_tracks + new_tracks:
            if track['ttl'] == 0:
                self.tracks_extendable.append(track)
            else:
                self.tracks_active.append(track)


        # update internal variables to be compatible with rest
        self.tracks = []
        for active, tt in zip([1,0,0],[self.tracks_active, self.tracks_extendable]):#, self.tracks_finished]):
            for i, tbox in enumerate(tt):
                steps_without_detection = self.ttl - tbox['ttl'] 
                self.tracks.append(Track(tbox['track_id'],tbox['bboxes'][-1],active=True,score=tbox['last_score'],history_length=self.config['ttl']))
                #score=0.5,history_length#,active,steps_without_detection,tbox['bboxes'],tbox['last_score']))#self.last_means[i],matched_track_scores[i]))
            

    '''# finish all remaining active and extendable tracks
    tracks_finished = tracks_finished + \
                    [track for track in tracks_active + tracks_extendable
                    if track['max_score'] >= sigma_h and track['det_counter'] >= t_min]

    # remove last visually tracked frames and compute the track classes
    for track in tracks_finished:
        if ttl != track['ttl']:
            track['bboxes'] = track['bboxes'][:-(ttl - track['ttl'])]
        track['class'] = max(set(track['classes']), key=track['classes'].count)

        del track['visual_tracker']
    '''
    