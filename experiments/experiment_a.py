import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_a(args, max_steps = 30000):
    print('[*] starting experiment A: keypoint estimation test loss under different number of training samples (100%, 50%, 10%, 1%)')
    config = model.get_config(args.project_id)
    config['train_video_ids'] = args.train_video_ids
    config['test_video_ids'] = args.test_video_ids
    config['experiment'] = 'A'
    config['kp_train_loss'] = 'focal'
    config['kp_test_losses'] = ['focal'] #['cce','focal']
    config['kp_max_steps'] = max_steps
    config['early_stopping'] = False
    config['kp_lr'] = 1e-4
    config['batch_size'] = 4
    #config['kp_lr'] = 2e-5

    #for data_ratio in [0.01,0.1,0.5,1.0][::-1]:
    for data_ratio in [0.01,0.1,0.5,1.0]:
        print('[*] starting sub experiment with %i/100 of data used' % int( 100. * data_ratio ))
        config['data_ratio'] = data_ratio
        print(config,'\n')
        checkpoint_path = roi_segm.train(config)

        clear_session()
        print(10*'\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=False,type=int)
    parser.add_argument('--test_video_ids',required=False,type=str)
    parser.add_argument('--train_video_ids',required=True,type=str)
    args = parser.parse_args()
    experiment_a(args)