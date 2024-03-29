# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import enum
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from multitracker.tracking.deep_sort.deep_sort import nn_matching

class DeepSORTTracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, config): # default 0.7 ; 30; 3 ; 0.2
        default_parameters = {
            'max_iou_distance': 0.5, 
            'max_age': 30, 
            'n_init': 3,
            'max_cosine_distance': 0.02, # Gating threshold for cosine distance metric (object appearance).
            'nn_budget': None # Maximum size of the appearance descriptors gallery. If None, no budget is enforced.
        }
        self.config = config 
        for k in default_parameters.keys():
            if not k in self.config:
                self.config[k] = default_parameters[k]

        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.config['max_cosine_distance'], self.config['nn_budget'])
        #self.metric = nn_matching.NearestNeighborDistanceMetric("euclidean", self.config['max_cosine_distance'], self.config['nn_budget'])
        
        
        self.max_iou_distance = self.config['max_iou_distance']
        self.max_age = self.config['max_age']
        self.n_init = self.config['n_init']

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def step(self, ob):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        [detections, detected_boxes, scores, features] = ob['detections']
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        if 0:
            print(f"\n{ob['frame_idx']} [*] matched {len(matches)} tracks")
            print(f"dets {len(detections)} tracks {len(self.tracks)}")
            print('unmatched_detections',unmatched_detections)
        if 0:
            for i,track in enumerate(self.tracks[-12:]):
                print('track',track.track_id,np.int32(track.to_tlwh()),"" if len(track.features)==0 else track.features[-1])
            for i,det in enumerate(detections):
                print('det',i,np.int32(det.tlwh),'features', det.feature)
                
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            print(ob['frame_idx'], '[*] created track',self._next_id,'for detection', detections[detection_idx].tlwh,detections[detection_idx].feature.shape)
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            if 0:
                print('feature distance')
                for y in range(cost_matrix.shape[0]):
                    for x in range(cost_matrix.shape[1]):
                        print('track', tracks[y].track_id, 'det', x, cost_matrix[y][x])
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        if 0:
            if len(unmatched_detections) > 0:
                for t in self.tracks[-6:]:
                    print('track', t.track_id, t.to_tlwh())
                for d in unmatched_detections:
                    print('unmatched det', detections[d].tlwh)
                print('self.max_iou_distance',self.max_iou_distance)
                print()
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        ## because we know the number of animals visible is fixed, 
        ## we can hard assign unmatched tracks to unmatched detections 
        ## and delete the other unmatched detections
        
        
        if 0:
            if len(unmatched_tracks) > 0: print('unmatched_tracks',unmatched_tracks)
            if len(unmatched_detections) > 0: print('unmatched_detections',unmatched_detections)
            for tt in unmatched_tracks:
                print('unmatched track',self.tracks[tt].mean)
            for tt in unmatched_detections:
                print('unmatched detection', detections[tt].tlwh, detections[tt].confidence)

            print('tracks',len(self.tracks),'new id',self._next_id)

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
