"""
batch size 8


batch size 4
    Faster R-CNN : 254.97011608547638 ms
    SSD : 20.89227835337321 ms

batch size 1
    Faster R-CNN : 322.45272530449756 ms
    SSD : 43.092139561971024 ms

"""

import numpy as np 
import os 
import time 
import json
import tensorflow as tf 
from glob import glob 
from tensorflow.keras.backend import clear_session
from multitracker.object_detection import finetune
from multitracker.tracking import inference

video_id = 13


def experiment_f_speed(checkpoint_base_dir):
    durations = {}
    cnt = 0
    
    experiment_dirs = glob(checkpoint_base_dir + '/*/')
    experiment_dirs = sorted(experiment_dirs)

    
    experiment_names = ['Faster R-CNN','SSD']
    batch_size = 8
    for i, experiment_dir in enumerate(experiment_dirs):
        print(i, experiment_dir, experiment_names[i])

        with open(os.path.join(experiment_dir, 'config.json')) as json_file:
            config = json.load(json_file)
        config['object_detection_batch_size'] = batch_size
        frame_bboxes, data_train, data_test = finetune.get_bbox_data(config, '14')
        #dataset = roi_segm.load_roi_dataset(config, mode='test', video_id = video_id)
        
        config['objectdetection_model'] = experiment_dir
        detection_model = finetune.load_trained_model(config)
    
        durations[experiment_names[i]] = 0.0
        for _ in range(3):
            for _, ims in data_test:
                _ = inference.detect_bounding_boxes(config, detection_model, ims)

        for _, ims in data_test:
            t1 = time.time()
            boxes = inference.detect_bounding_boxes(config, detection_model, ims)
            durations[experiment_names[i]] += time.time() - t1 
            cnt += ims.shape[0]

        print(experiment_names[i],':', durations[experiment_names[i]]/cnt *1000. ,'ms')
        clear_session()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_base_dir',required=True)
    args = parser.parse_args()
    experiment_f_speed(args.checkpoint_base_dir)