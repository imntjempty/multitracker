import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_c(args, max_steps = 50000):
    print('[*] starting experiment C: keypoint estimation test loss/inference speed: EfficientNet vs VGG16')
    config = model.get_config(args.project_id)
    model.create_train_dataset(config)
    config['video_id'] = int(args.video_id)

    config['experiment'] = 'C'
    config['mixup']=False
    config['cutmix'] = False
    config['kp_hflips']=False 
    config['kp_vflips']=False 
    config['kp_rot90s'] = False
    config['kp_blurpool'] = False

    config['train_loss'] = 'focal'
    config['test_losses'] = ['focal'] #['cce','focal']
    config['max_steps'] = max_steps
    #config['max_steps'] = 15000
    config['early_stopping'] = False
    config['kp_rotation_augmentation'] = bool(0)
    config['lr'] = 1e-4
    config['kp_num_hourglass'] = 1
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
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    experiment_c(args)