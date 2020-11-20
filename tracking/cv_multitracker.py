"""
    implement OpenCV multi tracker instead of DeepSort
    https://www.pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/


    simple idea:
        setup by init tracker for each detection of first frame
        a) for each frame always first update old tracker
        b) then find for each tracker detection with highest iou (at least 0.5)
        c) if prediction matched with detection:
              new general position is average of tracker prediction and matched detection
              replace tracker in multilist by init again with new general bounding box
              steps_without_detection = 0
              set active
        d) else:
              new general position is tracker prediction
              steps_without_detection++
              if steps_without_detection>thresh: set inactive
        e) reassign: for each unmatched detection
            if len(active tracks)==fixed_number:
                calc center distance to all inactive tracks, merge with nearest track
            else:
                add new track
"""

import argparse
import time
import cv2 as cv 
import numpy as np 
from multitracker.experiments.roi_curve import calc_iou

def tlbr2tlhw(tlbr):
    return [tlbr[0],tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1]]
def tlhw2tlbr(tlhw):
    return [tlhw[0],tlhw[1], tlhw[0]+tlhw[2],tlhw[1]+tlhw[3]]
def tlhw2chw(tlhw):
    return [ tlhw[0]+tlhw[2]/2. , tlhw[1]+tlhw[3]/2., tlhw[2],tlhw[3] ]

class OpenCVTrack(object):
    def __init__(self,track_id,tlhw,active,steps_without_detection,last_means):
        self.tlhw,self.active,self.steps_without_detection = tlhw,active,steps_without_detection
        self.track_id, self.time_since_update = track_id, steps_without_detection
        
        self.last_means = last_means
        self.last_means = [ tlhw2chw(p) for p in self.last_means ]
        
    def is_confirmed(self):
        return self.active 
    def to_tlbr(self): 
        return tlhw2tlbr(self.tlhw)
    def to_tlwh(self):
        return self.tlhw

class OpenCVMultiTracker(object):
    def __init__(self, fixed_number):
        self.fixed_number = fixed_number
        self.thresh_set_inactive = 10
        self.thresh_set_dead = 100
        self.maximum_other_track_init_distance = 350
        self.maximum_nearest_inactive_track_distance = self.maximum_other_track_init_distance * 4

        self.tracks = []
        self.trackers = cv.MultiTracker_create()
        #self.tracker_func = cv.TrackerCSRT_create
        self.active = [False for _ in range(fixed_number)]
        self.alive = [False for _ in range(fixed_number)]
        self.active_cnt = 0
        self.steps_without_detection = [0 for _ in range(fixed_number)]
        self.last_means = [[] for _ in range(fixed_number)]

        print('[*] inited OpenCVMultiTracker with fixed number %s'%fixed_number)

    def step(self,ob):
        debug = True 
        frame = ob['img']
        [detected_boxes,scores, features] = ob['detections']
    
        # a) update all trackers of last frame
        (success, tracked_boxes) = self.trackers.update(frame)
        
        # b) then find for each tracker unmatched detection with highest iou (at least 0.5)
        matched_detections = [False for _ in detected_boxes]
        matched_tracks = [False for _ in range(self.fixed_number)]
        #print('tracked_boxes',len(tracked_boxes))
        #print('detected_boxes',len(detected_boxes))

        for i,tbox in enumerate(self.trackers.getObjects()):
            (x, y, w, h) = [int(v) for v in tbox]
            highest_iou, highest_iou_idx = -1.,-1
            for j, dbox in enumerate(detected_boxes):
                if not matched_detections[j]:
                    iou = calc_iou(tlhw2tlbr(tbox),tlhw2tlbr(dbox))
                    if iou > highest_iou:
                        highest_iou, highest_iou_idx = iou, j 
                
            #print(i,highest_iou,highest_iou_idx,'coords',x,y,w,h)
            # if prediction matched with detection:
            if highest_iou > 0.5:
                # new general position is average of tracker prediction and matched detection
                alpha = 0.5
                box = alpha * tbox + (1.-alpha) * detected_boxes[highest_iou_idx]
                self.last_means[i].append(box)
                
                # replace tracker in multilist by init again with new general bounding box
                obs = self.trackers.getObjects()
                obs[i] = box
                self.trackers = cv.MultiTracker_create()
                for ob in obs:
                    self.trackers.add(cv.TrackerCSRT_create(), frame, tuple([int(cc) for cc in ob]))
                if debug: print('[*]   updated active tracker %i with detection %i' % (i,highest_iou_idx))
                
                self.steps_without_detection[i] = 0
                if not self.active[i]:
                    self.active[i] = True
                    self.alive[i] = True
                    self.active_cnt += 1
                matched_tracks[i] = True
                matched_detections[highest_iou_idx] = True
            else:
                #print(i,'steps_without_detection',self.steps_without_detection)
                self.last_means[i].append(tbox)
                self.steps_without_detection[i] += 1
                if self.steps_without_detection[i] > self.thresh_set_inactive and self.active[i]:
                    self.active[i] = False
                    self.active_cnt -= 1 
                if self.steps_without_detection[i] > self.thresh_set_dead and self.alive[i]:
                    self.alive[i] = False 

        
        # e) reassign: check if unmatched detection is a new track or gets assigned to inactive track (depending on barrier fixed number)
        for j in range(min(len(detected_boxes),self.fixed_number)):
            dbox = detected_boxes[j] # boxes are sorted desc as scores!
            if not matched_detections[j]: 
                if self.active_cnt < self.fixed_number:
                    # check if appropiate minimum distance to other track before initiating
                    other_track_near = False 
                    for tbox in self.trackers.getObjects():
                        other_track_near = other_track_near or np.linalg.norm(dbox-tbox) < self.maximum_other_track_init_distance
                            
                    if not other_track_near:
                        # add new track
                        dboxi = tuple([int(cc) for cc in dbox])
                        self.trackers.add(cv.TrackerCSRT_create(), frame, dboxi)
                        self.active[self.active_cnt] = True 
                        self.alive[self.active_cnt] = True
                        if debug: print('[*]   added tracker %i with detection %i' % (self.active_cnt,j))
                        self.active_cnt += 1
                else:
                    # calc center distance to all inactive tracks, merge with nearest track
                    nearest_inactive_track_distance, nearest_inactive_track_idx = 1e7,-1
                    for i, tbox in enumerate(self.trackers.getObjects()):
                        if not matched_tracks[i]:
                            (detx, dety, detw, deth) = dbox 
                            (trackx,tracky,trackw,trackh) = tbox
                            dist = np.sqrt(((detx+detw/2.)-(trackx+trackw/2.))**2 + ((dety+deth/2.) -(tracky+trackh/2.))**2 )
                            if nearest_inactive_track_distance > dist:
                                nearest_inactive_track_distance, nearest_inactive_track_idx = dist, i
                    
                    # merge by initing tracker with this detected box
                    # replace tracker in multilist by init again with new general bounding box
                    obs = self.trackers.getObjects()
                    #if nearest_inactive_track_idx >= 0:
                    if nearest_inactive_track_distance < self.maximum_nearest_inactive_track_distance:
                        matched_detections[j] = True
                        matched_tracks[nearest_inactive_track_idx] = True
                        obs[nearest_inactive_track_idx] = detected_boxes[j]
                        self.active[nearest_inactive_track_idx] = True
                        self.alive[nearest_inactive_track_idx] = True 
                        self.active_cnt += 1 
                        self.steps_without_detection[nearest_inactive_track_idx] = 0
                        self.trackers = cv.MultiTracker_create()
                        for ob in obs:
                            self.trackers.add(cv.TrackerCSRT_create(), frame, tuple([int(cc) for cc in ob]))
                        if debug: print('[*]   updated inactive tracker %i with detection %i' % (nearest_inactive_track_idx,j))

        # update internal variables to be compatible with rest
        self.tracks = []
        for i, tbox in enumerate(self.trackers.getObjects()):
            self.tracks.append(OpenCVTrack(i,tbox,self.active[i],self.steps_without_detection[i],self.last_means[i]))
            
        if debug:
            for i, tbox in enumerate(self.trackers.getObjects()):
                if len(self.last_means[i])>0:
                    print('Tracker',i,self.last_means[i][-1],'active',self.active[i],'steps misses',self.steps_without_detection[i])
            for j, dbox in enumerate(detected_boxes):
                print('Detect',j,str(scores[j])[1:4],dbox)
            print()

def example(file_video):
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv.TrackerCSRT_create,
        "kcf": cv.TrackerKCF_create,
        "boosting": cv.TrackerBoosting_create,
        "mil": cv.TrackerMIL_create,
        "tld": cv.TrackerTLD_create,
        "medianflow": cv.TrackerMedianFlow_create,
        "mosse": cv.TrackerMOSSE_create
    }
    tracker_name = "csrt"

    # initialize OpenCV's special multi-object tracker
    trackers = cv.MultiTracker_create()

    video_reader = cv.VideoCapture(file_video)

    while video_reader.isOpened():
        ret, frame = video_reader.read()
        frame = cv.resize(frame,None,None,fx=0.5,fy=0.5)

        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        (success, boxes) = trackers.update(frame)
        # loop over the bounding boxes and draw then on the frame
        for i,box in enumerate(boxes):
            (x, y, w, h) = [int(v) for v in box]
            c = 255 // (i+1)
            color = (255-c, c, 0)
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # show the output frame
        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            box = cv.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
            # create a new object tracker for the bounding box and add it
            # to our multi-object tracker
            tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
            trackers.add(tracker, frame, box)
    
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
    
    video_reader.release()
    # close all windows
    cv.destroyAllWindows()


if __name__ == '__main__':
    file_video = '/home/alex/data/multitracker/projects/7/videos/from_above_Oct2020_2_12fps.mp4'
    example(file_video)