"""
    implementation of Tracktor++ 
    Tracking without bells and whistles
    https://arxiv.org/pdf/1903.05625v3.pdf

    ported from official pytorch implementation https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py
"""

import os 
import numpy as np 


class Tracker(object):
    def __init__(self):
        self.cl = 1 # only one class
        self.tracks = []
		self.inactive_tracks = []

        self.track_num = 0
        self.results = {}
        self.im_index = 0

    def tracks_to_inactive(self, tracks):
		self.tracks = [t for t in self.tracks if t not in tracks]
		for t in tracks:
			t.pos = t.last_pos[-1]
		self.inactive_tracks += tracks

    def predict(self):
        pass # legacy function from DeepSORT
    
    def update(self, detections):

        # apply motion model

class Track(object):
    self.last_detections = [] 
    self.last_means = [] 
    
    def __init__(self):

        self.last_detections.append(detection)
        self.last_means.append(self.mean)