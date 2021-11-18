
from copy import deepcopy
import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..utils import TrackEvalException
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing


class Multitracker(_BaseDataset):
    """Dataset class for Multitracker tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': os.path.join(code_path, 'data/gt/bdd100k/bdd100k_val'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/bdd100k/bdd100k_val'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle'],
            # Valid: ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle']
            'SPLIT_TO_EVAL': 'val',  # Valid: 'training', 'val',
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        print('config',config)
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        print('self.config',self.config)
        self.tracker_list = ['tr4cker']
        self.seq_list = ['seg1']
        self.class_list = ['mouse']
        self.class_name_to_class_id = {self.class_list[0]:1}

        self.output_fol = os.path.expanduser('~/data/multitracker/evaluation')
        self.output_sub_fol = 'seg1'


    def get_display_name(self, tracker):
        return "multitracker" #self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the BDD100K format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        
        csv_gt = '/home/alex/data/multitracker/projects/1/1/trackannotation.csv'
        csv_tracked = '/home/alex/data/multitracker/projects/1/tracking_Tracking_under_occlusion_UpperBound_ssd_2stack_none_2020-11-25_08-47-15_22772819_rec-00.00.00.000-00.10.20.916-seg1.csv'

        raw_data = {'tracker_ids':{},'tracker_classes':{},'tracker_dets':{},'gt_ids':{},'gt_classes':{},'gt_dets':{}}
        ## read and parse ground truth CSV
        with open(csv_gt, 'r') as f:
            lines_gt = [l.replace('\n','').split(',') for l in f.readlines()]
            cnt_gt = 0 
            for [frame_idx,idv,x1,y1,x2,y2] in lines_gt:
                if cnt_gt > 0:
                    [frame_idx,idv,x1,y1,x2,y2] = [float(s) for s in [frame_idx,idv,x1,y1,x2,y2]]
                    frame_idx,idv = int(frame_idx),int(idv)
                    for k in ['gt_ids','gt_classes','gt_dets','tracker_ids','tracker_classes','tracker_dets']:
                        if frame_idx not in raw_data[k]:
                            raw_data[k][frame_idx] = []
                    raw_data['gt_ids'][frame_idx].append(idv)
                    raw_data['gt_classes'][frame_idx].append(1) # only one class
                    raw_data['gt_dets'][frame_idx].append([x1,y1,x2,y2])
                cnt_gt += 1 
        
        ## read and parse tracked CSV
        with open(csv_tracked, 'r') as f:
            lines_tracked = [l.replace('\n','').split(',') for l in f.readlines()]
            cnt_tracked = 0 
            for [video_id,frame_idx,idv,centerx,centery,x1,y1,w,h,time_since_update] in lines_tracked:
                if cnt_tracked > 0:
                    [frame_idx,idv,x1,y1,w,h] = [float(s) for s in [frame_idx,idv,x1,y1,w,h]]
                    x2 = x1 + w 
                    y2 = y1 + h 
                    frame_idx,idv = int(frame_idx),int(idv)
                    #for k in ['gt_ids','gt_classes','gt_dets','tracker_ids','tracker_classes','tracker_dets']:
                    #    if frame_idx not in raw_data[k]:
                    #        raw_data[k][frame_idx] = []
                    if frame_idx in raw_data['gt_ids']:
                        raw_data['tracker_ids'][frame_idx].append(idv)
                        raw_data['tracker_classes'][frame_idx].append(1) # only one class
                        raw_data['tracker_dets'][frame_idx].append([x1,y1,x2,y2])
                cnt_tracked += 1 

        _raw = {}
        for k in raw_data.keys():
            #print(k,list(raw_data[k].keys())[0],list(raw_data[k].keys())[-1])
            #print()
            _raw[k] = []
            for frame_idx in raw_data[k].keys():
                _raw[k].append(raw_data[k][frame_idx])
            _raw[k] = np.array(_raw[k])

        raw_data = _raw 


        raw_data['num_timesteps'] = min(cnt_gt, cnt_tracked) -1
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.

        """
        print('get_preprocessed_seq_data',cls,'ts',raw_data['num_timesteps'])
        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        #data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        data = {}
        for k in data_keys:
            data[k] = []
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls)
            if 0:
                gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id)
                gt_class_mask = gt_class_mask.astype(np.bool)
                gt_ids = raw_data['gt_ids'][t][gt_class_mask]
                gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            
                tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id)
                tracker_class_mask = tracker_class_mask.astype(np.bool)
                tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
                tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
                similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

                # Match tracker and gt dets (with hungarian algorithm)
                unmatched_indices = np.arange(tracker_ids.shape[0])
                if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                    matching_scores = similarity_scores.copy()
                    matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                    match_rows, match_cols = linear_sum_assignment(-matching_scores)
                    actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                    match_cols = match_cols[actually_matched_mask]
                    unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)
            else:
                if t < len(raw_data['gt_ids']):
                    gt_ids = np.array(raw_data['gt_ids'][t])
                    gt_dets = np.array(raw_data['gt_dets'][t])
                    tracker_ids = np.array(raw_data['tracker_ids'][t])
                    tracker_dets = np.array(raw_data['tracker_dets'][t])
                    #print('gt_ids',gt_ids,'gt_dets',gt_dets,'tracker_ids',tracker_ids,'tracker_dets',tracker_dets)
                    similarity_scores = raw_data['similarity_scores'][t]
                    #print('similarity_scores',similarity_scores)
                    if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                        match_rows, match_cols = linear_sum_assignment(-similarity_scores)
                        for r, c in zip(match_rows, match_cols):
                            print('match',r,c)

            '''data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores'''
            if len(gt_ids)>0 and len(tracker_ids)>0:
                data['gt_ids'].append(gt_ids)
                data['gt_dets'].append(gt_dets)
                data['tracker_ids'].append(tracker_ids)
                data['tracker_dets'].append(tracker_dets)
                data['similarity_scores'].append(similarity_scores)

                print('t',t,'trackids',len(data['tracker_ids']),len(data['gt_ids']),'qqq')#,data['gt_ids'][3])
                #if t in data['gt_ids']:
                unique_gt_ids += list(np.unique(data['gt_ids'][-1]))
                #if t in data['tracker_ids']:
                unique_tracker_ids += list(np.unique(data['tracker_ids'][-1]))
                #if t in data['tracker_ids']:
                num_tracker_dets += len(data['tracker_ids'][-1])
                #if t in data['gt_ids']:
                num_gt_dets += len(data['gt_ids'][-1])

        for k in data_keys:
            #data[k] = np.array(data[k])
            for i in range(len(data[k])):
                data[k][i] = np.array(data[k][i])
        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']

        # Ensure that ids are unique per timestep.
        #self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        #print('gt_dets_t',gt_dets_t)
        #print('tracker_dets_t',tracker_dets_t)
        gt_dets_t = np.array(gt_dets_t)
        tracker_dets_t = np.array(tracker_dets_t)
        if len(gt_dets_t) > 0 and len(tracker_dets_t) > 0:
            similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='x0y0x1y1')
            return similarity_scores
        else:
            return [[]]
