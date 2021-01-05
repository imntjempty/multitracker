import numpy as np 
import os 
import tensorflow as tf 
from copy import deepcopy
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_b(args, max_steps = 12000):
    print('[*] starting experiment B: keypoint estimation test loss: pretrained vs random init network')
    config = model.get_config(args.project_id)
    config['train_video_ids'] = args.train_video_ids
    config['test_video_ids'] = args.test_video_ids
    
    config['experiment'] = 'B'
    config['kp_backbone'] = 'hourglass2'
    config['kp_train_loss'] = 'focal'
    config['kp_test_losses'] = ['focal'] #['cce','focal']
    config['kp_max_steps'] = max_steps
    #config['kp_max_steps'] = 15000
    config['early_stopping'] = False
    config['batch_size'] = 8
    config['kp_lr'] = 1e-4

    config_b = deepcopy(config)
    #for should_init_pretrained in [False, True]:
    for should_init_pretrained in [False,True]:
        #for data_ratio in [0.01,1.0]:
        for augm in [False,True]:
            print('[*] starting sub experiment %s weight initialization' % ['without','with'][int(should_init_pretrained)])
            #config['data_ratio'] = data_ratio
            if not augm:
                config_b['kp_mixup'] = False 
                config_b['kp_cutmix'] = False 
                config_b['kp_hflips'] = False 
                config_b['kp_vflips'] = False 
                config_b['kp_rotation_augmentation'] = False 
                config_b['kp_rot90s'] = False 
                config_b['kp_im_noise'] = False 
                config_b['kp_im_transf'] = False 
            config_b['should_init_pretrained'] = should_init_pretrained
            print(config_b,'\n')
            checkpoint_path = roi_segm.train(config_b, log_images=False)
            
        clear_session()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--test_video_ids',required=True,type=str)
    parser.add_argument('--train_video_ids',required=True,type=str)
    args = parser.parse_args()
    experiment_b(args)