import numpy as np 
import os 
import tensorflow as tf 
from datetime import datetime 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm
from multitracker.object_detection.finetune import finetune

def experiment_e(args, train_video_ids = None):
    print('[*] starting experiment E: object detection test loss under different number of training samples')
    config = model.get_config(args.project_id)
    config['video_id'] = int(args.video_id)
    if train_video_ids is not None:
        config['train_video_ids'] = train_video_ids
    else:
        config['train_video_ids'] = args.train_video_ids
    config['test_video_ids'] = args.test_video_ids
    config['experiment'] = 'E'
    config['maxsteps_objectdetection'] = 50000
    #config['kp_max_steps'] = 15000
    config['early_stopping'] = False
    config['finetune'] = False 
    '''config['object_augm_flip'] = bool(0)
    config['object_augm_rot90'] = bool(0)
    config['object_augm_gaussian'] = bool(0)
    config['object_augm_image'] = bool(0)
    config['object_augm_mixup'] = bool(0)
    config['object_augm_crop'] = bool(0)
    config['object_augm_stitch'] = bool(0)'''

    #for data_ratio in [1.0]:
    for data_ratio in [0.01,0.1,0.5,1.0]:
        now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
        print('[*] starting sub experiment with %i/100 of data used' % int( 100. * data_ratio ))
        config['data_ratio'] = data_ratio
        checkpoint_dir = os.path.expanduser('~/checkpoints/experiments/%s/E/%i-%s' %(config['project_name'], int( 100. * data_ratio ), now))
        config['maxsteps_objectdetection'] = {0.01: 20000, 0.1: 20000, 0.5: 30000, 1.0: 30000}[data_ratio]
    
        print(config,'\n')
        finetune(config, checkpoint_dir)

        clear_session()
        print(10*'\n')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--test_video_ids',required=True,type=str)
    parser.add_argument('--train_video_ids',required=True)
    args = parser.parse_args()
    experiment_e(args)