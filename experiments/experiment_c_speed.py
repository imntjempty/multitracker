
"""
batch size 8
    Efficient U-Net: 10.76847409445142 ms
    Hourglass 2:      4.709767916845897 ms
    Hourglass 4:      5.631455668696651 ms
    Hourglass 8:      8.231820568205817 ms
    PSP :             0.8209213377937438 ms
    VGG U-Net :       1.1135049597926872 ms
batch size 4
    Efficient U-Net : 17.949864977882022 ms
    Hourglass 2 : 8.884322075616746 ms
    Hourglass 4 : 10.598163756113204 ms
    Hourglass 8 : 15.01116298493885 ms
    PSP : 1.5972614288330078 ms
    VGG U-Net : 1.2325969322648629 ms
batch size 1
    Efficient U-Net : 63.83616962130108 ms
    Hourglass 2 : 33.35623703305683 ms
    Hourglass 4 : 39.810869428846566 ms
    Hourglass 8 : 56.879131566910516 ms
    PSP : 6.385420239160932 ms
    VGG U-Net : 2.5761285156169267 ms
"""
import numpy as np 
import os 
import time 
import json
import tensorflow as tf 
from glob import glob 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm, unet
from multitracker.tracking import inference

video_id = 13

def experiment_c_speed(checkpoint_base_dir):
    durations = {}
    cnt = 0
    
    experiment_dirs = glob(checkpoint_base_dir + '/*/')
    experiment_dirs = sorted(experiment_dirs)

    
    experiment_names = ['Efficient U-Net', 'Hourglass 2', 'Hourglass 4', 'Hourglass 8', 'PSP', 'VGG U-Net']
    batch_size = 1
    for i, experiment_dir in enumerate(experiment_dirs):
        #print(i, experiment_dir, experiment_names[i])

        with open(os.path.join(experiment_dir, 'config.json')) as json_file:
            config = json.load(json_file)
        config['batch_size'] = batch_size
        dataset = roi_segm.load_roi_dataset(config, mode='test', video_id = video_id)
        
        net = inference.load_keypoint_model(experiment_dir)
        durations[experiment_names[i]] = 0.0
        for _ in range(3):
            for x,y in dataset:
                y_ = net(x)
                
        for x,y in dataset:
            #print('x',x.shape)
            t_inf_start = time.time()
            y_ = net(x)
            durations[experiment_names[i]] += time.time() - t_inf_start
            cnt += x.shape[0]

        print(experiment_names[i],':', durations[experiment_names[i]]/cnt *1000. ,'ms')
        clear_session()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_base_dir',required=True)
    args = parser.parse_args()
    experiment_c_speed(args.checkpoint_base_dir)