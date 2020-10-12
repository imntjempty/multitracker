import numpy as np 
import os 
import tensorflow as tf 
from datetime import datetime 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm
from multitracker.object_detection.finetune import finetune

def experiment_e(args):
    print('[*] starting experiment E: object detection test loss under different number of training samples')
    config = model.get_config(args.project_id)
    config['video_id'] = int(args.video_id)
    config['experiment'] = 'E'
    config['max_steps'] = 50000
    #config['max_steps'] = 15000
    config['early_stopping'] = False
    
    for data_ratio in [0.01,0.1,0.5,1.0]:
        now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
        print('[*] starting sub experiment with %i/100 of data used' % int( 100. * data_ratio ))
        config['data_ratio'] = data_ratio
        checkpoint_dir = os.path.expanduser('~/checkpoints/experiments/%s/E/%i-%s' %(config['project_name'], int( 100. * data_ratio ), now))
        print(config,'\n')
        finetune(config, checkpoint_dir)

        clear_session()
        print(10*'\n')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    experiment_e(args)