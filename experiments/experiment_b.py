import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_b(args, max_steps = 30000):
    print('[*] starting experiment B: keypoint estimation test loss: pretrained vs random init network')
    config = model.get_config(args.project_id)
    config['train_video_ids'] = args.train_video_ids
    config['video_id'] = int(args.video_id)
    model.create_train_dataset(config)

    config['experiment'] = 'B'
    '''config['kp_mixup'] = False
    config['kp_cutmix'] = False
    config['kp_hflips'] = False 
    config['kp_vflips'] = False 
    config['kp_rot90s'] = False
    config['kp_rotation_augmentation'] = bool(0)
    config['kp_blurpool'] = False'''
    config['kp_backbone'] = 'hourglass2'
    config['kp_train_loss'] = 'focal'
    config['kp_test_losses'] = ['focal'] #['cce','focal']
    config['kp_max_steps'] = max_steps
    #config['kp_max_steps'] = 15000
    config['early_stopping'] = False
    config['batch_size'] = 4
    config['kp_lr'] = 1e-4

    #for should_init_pretrained in [False, True]:
    for should_init_pretrained in [False]:
        print('[*] starting sub experiment %s weight initialization' % ['without','with'][int(should_init_pretrained)])
        config['should_init_pretrained'] = should_init_pretrained
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
    experiment_b(args)