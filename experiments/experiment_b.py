import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_b():
    print('[*] starting experiment B: keypoint estimation test loss: pretrained vs random init network')
    config = model.get_config(args.project_id)
    model.create_train_dataset(config)
    config['video_id'] = int(args.video_id)

    for should_init_pretrained in [False, True]:
        print('[*] starting sub experiment %s weight initialization' % ['without','with'][int(should_init_pretrained)])
        config['experiment'] = 'B'
        config['should_init_pretrained'] = should_init_pretrained
        print(config,'\n')
        checkpoint_path = roi_segm.train(config)
        
        clear_session()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    experiment_b(args)