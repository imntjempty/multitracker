import numpy as np 
import os 
import tensorflow as tf 
from datetime import datetime 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm
from multitracker.object_detection.finetune import finetune

def experiment_g(args, train_video_ids = None):
    print('[*] starting experiment G: object detection pretrained vs random init network, w/wo augmentation')
    config = model.get_config(args.project_id)
    
    config['data_dir'] = '/home/alex/data/multitracker'
    if train_video_ids is not None:
        config['train_video_ids'] = train_video_ids
    else:
        config['train_video_ids'] = args.train_video_ids
    config['test_video_ids'] = args.test_video_ids
    config['experiment'] = 'G'
    config['early_stopping'] = False
    config['object_finetune_warmup'] = 1000
    config['maxsteps_objectdetection'] = 20000
    config['lr_objectdetection'] = 0.005 
    config['object_augm_gaussian'] = bool(0)
    config['object_augm_image'] = bool(0)
    config['object_augm_mixup'] = bool(0)
    
    #for should_init_pretrained in [True]:
    #    for augm in [True]:
    for should_init_pretrained in [False,True]:
        for augm in [False,True]:
            now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
            if not augm:
                config['object_augm_flip'] = bool(0)
                config['object_augm_rot90'] = bool(0)
            else:
                config['object_augm_flip'] = bool(1)
                config['object_augm_rot90'] = bool(1)
            config['object_pretrained'] = should_init_pretrained    
            str_augm = 'augm' if augm else 'noaugm'
            str_pretrained = 'pretrained' if should_init_pretrained else 'scratch'
            print('[*] starting sub experiment',str_augm,str_pretrained)

            checkpoint_dir = os.path.expanduser('~/checkpoints/experiments/%s/G/%s-%s-%s' %(config['project_name'], str_augm, str_pretrained, now))
            #config['maxsteps_objectdetection'] = {0.01: 12000, 0.1: 15000, 0.5: 20000, 1.0: 25000}[data_ratio]
            print(finetune)
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
    experiment_g(args)