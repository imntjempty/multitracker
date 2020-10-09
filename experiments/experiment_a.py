import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_a(args):
    print('[*] starting experiment A: keypoint estimation test loss under different number of training samples (100%, 50%, 20%, 10%)')
    config = model.get_config(args.project_id)
    model.create_train_dataset(config)
    config['experiment'] = 'A'
    config['video_id'] = int(args.video_id)
    config['mixup']=False
    config['hflips']=False 
    config['vflips']=False 
    config['backbone'] = 'efficientnetLarge'
    config['train_loss'] = 'focal'
    config['test_losses'] = ['focal'] #['cce','focal']
    config['max_steps'] = 50000
    #config['max_steps'] = 15000
    config['early_stopping'] = False
    config['rotation_augmentation'] = bool(0)
    config['lr'] = 1e-4

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
    parser.add_argument('--video_id',required=False,type=int)
    args = parser.parse_args()
    experiment_a(args)