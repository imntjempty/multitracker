
import os
from glob import glob 
import subprocess
from collections import deque

import cv2 as cv 
import numpy as np

import tensorflow as tf 
assert tf.__version__.startswith('2.'), 'YOU MUST INSTALL TENSORFLOW 2.X'
print('[*] TF version',tf.__version__)
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import Model

from multitracker.tracking.deep_sort.application_util import preprocessing
from multitracker.tracking.deep_sort.application_util import visualization
from multitracker.tracking.deep_sort.deep_sort import nn_matching
from multitracker.tracking.deep_sort import deep_sort_app
from multitracker.tracking.deep_sort.deep_sort.detection import Detection
from multitracker.tracking.deep_sort.deep_sort.tracker import Tracker

from multitracker import autoencoder
from multitracker.keypoint_detection import roi_segm, unet
from multitracker.tracking.inference import get_heatmaps_keypoints
from multitracker.tracking.keypoint_tracking import tracker as keypoint_tracking
from multitracker.be import video
from multitracker import util 
from multitracker.experiments.roi_curve import calc_iou
from multitracker.tracking import inference

colors = util.get_colors()



"""
    implementation of Tracktor++ 
    Tracking without bells and whistles
    https://arxiv.org/pdf/1903.05625v3.pdf

    ported from official pytorch implementation to tf2 https://github.com/phil-bergmann/tracking_wo_bnw/blob/master/src/tracktor/tracker.py
"""



class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([np.array(pos)], maxlen=mm_steps + 1)
        self.last_v = None 
        self.gt_id = None

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = tf.concat(list(self.features), axis=0)
        else:
            features = self.features[0]
        features = tf.reduce_mean(features,axis=0, keepdims=True)
        dist = tf.reduce_sum(tf.squared_difference(features, test_features), 2)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())


class Tracker(object):
    def __init__(self,config):
        self.config = config 
        self.cl = 1 # only one class
        self.tracks = []
        self.inactive_tracks = []

        # always reset
        self.track_num = 0
        self.results = {}
        self.im_index = 0

        import yaml
        yaml_file = '/home/alex/github/multitracker/tracking/tracktor/config.yaml'
        with open(yaml_file) as f:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            tracker_cfg = yaml.load(f, Loader=yaml.FullLoader)['tracktor']['tracker']
            print('tracker_cfg',tracker_cfg)

        self.detection_person_thresh = tracker_cfg['detection_person_thresh']
        self.regression_person_thresh = tracker_cfg['regression_person_thresh']
        self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
        self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
        self.public_detections = tracker_cfg['public_detections']
        self.inactive_patience = tracker_cfg['inactive_patience']
        self.do_reid = tracker_cfg['do_reid']
        self.max_features_num = tracker_cfg['max_features_num']
        self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
        self.do_align = tracker_cfg['do_align']
        self.motion_model_cfg = tracker_cfg['motion_model']

        self.warp_mode = eval(tracker_cfg['warp_mode'])
        self.number_of_iterations = tracker_cfg['number_of_iterations']
        self.termination_eps = tracker_cfg['termination_eps']


    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add_track(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them."""
        self.tracks.append(Track(
            new_det_pos,
            new_det_scores,
            self.track_num + 1,
            new_det_features,
            self.inactive_patience,
            self.max_features_num,
            self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
        ))
        self.track_num += 1

    def regress_tracks(self, blob):
        """Regress the position of the tracks and also checks their scores."""
        pos = self.get_pos()

        # regress
        boxes, scores = self.obj_detect.predict_boxes(pos)
        pos = boxes #clip_boxes_to_image(boxes, blob['img'].shape[-2:])

        s = []
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                # t.prev_pos = t.pos
                t.pos = pos[i].view(1, -1)

        return tf.convert_to_tensor(np.array(s[::-1]))

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = tf.concat([t.pos for t in self.tracks], 0)
        else:
            pos = (0,0)
        return pos

    def get_features(self):
        """Get the features of all active tracks."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = tf.concatenate([t.features for t in self.tracks], 0)
        else:
            features = (0,0)
        return features

    def get_inactive_features(self):
        """Get the features of all inactive tracks."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = tf.concat([t.features for t in self.inactive_tracks], 0)
        else:
            features = (0,0)
        return features

    def reid(self, blob, new_det_pos, new_det_scores):
        print("""Tries to ReID inactive tracks with provided detections.""")
        
        return new_det_pos, new_det_scores,[]

    def get_appearances(self, blob):
        """Uses the siamese CNN to get the features for all active tracks."""
        new_features = self.reid_network.test_rois(blob['img'], self.get_pos()).data
        return new_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def align(self, blob):
        """Aligns the positions of active and inactive tracks depending on camera motion."""
        if self.im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
            im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
            im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, self.number_of_iterations,  self.termination_eps)
            cc, warp_matrix = cv.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
            warp_matrix = tf.convert_to_tensor(warp_matrix)

            for t in self.tracks:
                t.pos = warp_pos(t.pos, warp_matrix)
                # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

            if self.do_reid:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            if self.motion_model_cfg['enabled']:
                for t in self.tracks:
                    for i in range(len(t.last_pos)):
                        t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

    def motion_step(self, track):
        """Updates the given track's position by one step based on track.last_v"""
        if self.motion_model_cfg['center_only']:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
        else:
            track.pos = track.pos + track.last_v

    def motion(self):
        """Applies a simple linear motion model that considers the last n_steps steps."""
        for t in self.tracks:
            last_pos = list(t.last_pos)

            # avg velocity between each pair of consecutive positions in t.last_pos
            if self.motion_model_cfg['center_only']:
                vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
            else:
                vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

            t.last_v = tf.reduce_mean(tf.stack(vs), 0)
            self.motion_step(t)

        if self.do_reid:
            for t in self.inactive_tracks:
                if len(t.last_v) > 0:
                    self.motion_step(t)

    def step(self, blob):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        for t in self.tracks:
            # add current position to last_pos list
            t.last_pos.append(t.pos.clone())

        ###########################
        # Look for new detections #
        ###########################

        #self.obj_detect.load_image(blob['img'])

        boxes, scores, features = blob['detections']

        '''if len(boxes) > 0:
            #boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

            # Filter out tracks that have too low person score
            inds = [i for i, score in enumerate(scores) if score > self.detection_person_thresh]
            #inds = tf.cast(tf.math.greater(scores, self.detection_person_thresh),'int32').nonzero().view(-1)
        else:
            inds = []

        if len(inds) > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = []
            det_scores = []'''

        ##################
        # Predict tracks #
        ##################

        num_tracks = 0
        nms_inp_reg = []
        if len(self.tracks)>0:
            # align
            if 0 and self.do_align: ## no camera motion! so no 
                self.align(blob)

            # apply motion model
            if self.motion_model_cfg['enabled']:
                self.motion()
                self.tracks = [t for t in self.tracks if t.has_positive_area()]

            # regress
            person_scores = self.regress_tracks(blob)

            if len(self.tracks)>0:
                # create nms input

                # nms here if tracks overlap
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)

                self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

                if len(keep) > 0 and self.do_reid:
                        new_features = self.get_appearances(blob)
                        self.add_features(new_features)

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        
        if len(boxes) > 0:

            #new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)


            # for each detection, check if has overlap with a track box, if not add detection as new track
            for ii in range(len(boxes)):
                overlaps_track = False 
                for ij in range(len(self.tracks)):
                    t = self.tracks[ij]
                    if calc_iou(boxes[ii],t.to_tlbr())>0.5:
                        overlaps_track = True 
                if not overlaps_track:
                    self.add_track(boxes[ii], scores[ii], features[ii])

        ####################
        # Generate Results #
        ####################

        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.pos[0].numpy(), np.array([t.score])])

        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        self.im_index += 1
        self.last_image = blob['img'][0]

    def get_results(self):
        return self.results





    def predict(self):
        pass # legacy function from DeepSORT
    
    def update(self, detections):
        pass
        # apply motion model




def run(config, detection_model, encoder_model, keypoint_model, crop_dim, tracker = None):
    min_confidence = 0.7
    min_confidence_keypoints = 0.6
    nms_max_overlap = 1.

    if 'video' in config and config['video'] is not None:
        video_reader = cv.VideoCapture( config['video'] )
    else:
        video_reader = None 
    
    [Hframe,Wframe,_] = cv.imread(glob(os.path.join(os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'test'),'*.png'))[0]).shape
    
    video_file_out = inference.get_video_output_filepath(config)
    if os.path.isfile(video_file_out): os.remove(video_file_out)
    import skvideo.io
    video_writer = skvideo.io.FFmpegWriter(video_file_out, outputdict={
        '-vcodec': 'libx264',  #use the h.264 codec
        '-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }) 

    seq_info = deep_sort_app.gather_sequence_info(config)
    visualizer = visualization.Visualization(seq_info, update_ms=5)
    print('[*] writing video file %s' % video_file_out)
    
    ## initialize tracker for boxes and keypoints
    if tracker is None:
        tracker = Tracker(config)
    keypoint_tracker = keypoint_tracking.KeypointTracker()
    frame_directory = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'train')
    frame_idx = 0
    config['count'] = 0
    while video_reader.isOpened():
        ret, frame = video_reader.read()
    
        frame_idx += 1 
        config['count'] = frame_idx

        frame_, detections = inference.detect_frame_boundingboxes(config, detection_model, encoder_model, seq_info, frame, frame_idx)
        detections = [d for d in detections if d.confidence >= min_confidence]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        features = np.array([d.feature for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        #print('[*] found %i detections' % len(detections))
        # Update tracker
        tracker.step({'img':frame,'detections':[boxes, scores, features]})

        keypoints = inference.inference_keypoints(config, frame, detections, keypoint_model, crop_dim, min_confidence_keypoints)
        # update tracked keypoints with new detections
        tracked_keypoints = keypoint_tracker.update(keypoints)

        print('%i - %i detections. %i keypoints' % (config['count'],len(detections), len(keypoints)),[kp for kp in keypoints])
        out = deep_sort_app.visualize(visualizer, frame, tracker, detections, keypoint_tracker, keypoints, tracked_keypoints, crop_dim)
        video_writer.writeFrame(cv.cvtColor(out, cv.COLOR_BGR2RGB)) #out[:,:,::-1])
        
        if 1:
            cv.imshow("tracking visualization", cv.resize(out,None,None,fx=0.5,fy=0.5))
            cv.waitKey(5)
        # Store results.
        '''for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])'''

