import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_d(args):
    print('[*] starting experiment D: keypoint estimation categorical cross entropy / focal loss test trained with categorical cross entropy and focal loss')
    config = model.get_config(args.project_id)
    model.create_train_dataset(config)
    config['video_id'] = int(args.video_id)
    config['test_losses'] = ['focal','cce']

    config['experiment'] = 'D'
    config['mixup']=False
    config['hflips']=False 
    config['vflips']=False 
    config['train_loss'] = 'focal'
    config['test_losses'] = ['focal'] #['cce','focal']
    config['max_steps'] = 50000
    #config['max_steps'] = 15000
    config['early_stopping'] = False
    config['rotation_augmentation'] = bool(0)
    config['lr'] = 1e-4

    for loss_name in ['focal','cce']:
        print('[*] starting sub experiment loss function %s' % loss_name)
        config['train_loss'] = loss_name
        
        print(config,'\n')
        checkpoint_path = roi_segm.train(config)
        
        clear_session()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    experiment_d(args)