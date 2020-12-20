import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_c(args, max_steps = 30000):
    print('[*] starting experiment C: keypoint estimation test loss/inference speed: EfficientNet vs VGG16')
    config = model.get_config(args.project_id)
    config['train_video_ids'] = args.train_video_ids
    config['test_video_ids'] = args.test_video_ids
    
    config['experiment'] = 'C'
    config['kp_train_loss'] = 'focal'
    config['kp_test_losses'] = ['focal'] #['cce','focal']
    config['kp_max_steps'] = max_steps
    #config['kp_max_steps'] = 15000
    config['early_stopping'] = False
    config['kp_lr'] = 1e-4
    config['kp_num_hourglass'] = 1
    config['batch_size'] = 4
    #for backbone in ['hourglass2']:
    for backbone in ['vgg16','efficientnetLarge','psp','hourglass4','hourglass8']:
        print('[*] starting sub experiment backbone %s' % backbone)
        config['kp_backbone'] = backbone
        print(config,'\n')
        checkpoint_path = roi_segm.train(config)
        
        clear_session()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--test_video_ids',required=True,type=str)
    parser.add_argument('--train_video_ids',required=True,type=str)
    args = parser.parse_args()
    experiment_c(args)