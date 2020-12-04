# python3.7 -m multitracker.experiments.all --project_id 7 --video_id 13 --train_video_ids 9,14
import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_d(args, max_steps = 50000, train_video_ids = None):
    print('[*] starting experiment D: keypoint estimation categorical cross entropy / focal loss test trained with categorical cross entropy and focal loss')
    config = model.get_config(args.project_id)
    model.create_train_dataset(config)
    config['video_id'] = int(args.video_id)
    if train_video_ids is not None:
        config['train_video_ids'] = train_video_ids
    else:
        config['train_video_ids'] = args.train_video_ids
    config['kp_test_losses'] = ['focal','cce','l2']

    config['experiment'] = 'D'
    config['kp_mixup']=False
    config['kp_cutmix']=False
    config['kp_rot90s'] = False
    config['kp_hflips']=False 
    config['kp_vflips']=False 
    config['kp_blurpool'] = False
    config['max_steps'] = max_steps
    config['kp_backbone'] = 'hourglass2'
    #config['max_steps'] = 15000
    config['early_stopping'] = False
    config['kp_rotation_augmentation'] = bool(0)
    config['kp_lr'] = 1e-4

    #for loss_name in ['focal','cce','l2']:
    for loss_name in ['focal','cce','l2']:
        print('[*] starting sub experiment loss function %s' % loss_name)
        config['kp_train_loss'] = loss_name
        
        print(config,'\n')
        checkpoint_path = roi_segm.train(config)
        
        clear_session()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    parser.add_argument('--train_video_ids',required=True,type=str)
    args = parser.parse_args()
    experiment_d(args)