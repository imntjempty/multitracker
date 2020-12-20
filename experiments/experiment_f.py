
import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm
from multitracker.object_detection.finetune import finetune
from datetime import datetime

def experiment_f(args, train_video_ids = None):
    print('[*] starting experiment F: object detection test loss SSD vs FasterRCNN')
    config = model.get_config(args.project_id)
    if train_video_ids is not None:
        config['train_video_ids'] = train_video_ids
    else:
        config['train_video_ids'] = args.train_video_ids
    config['test_video_ids'] = args.test_video_ids
    config['experiment'] = 'F'
    config['maxsteps_objectdetection'] = 20000
    config['early_stopping'] = False
    config['finetune'] = False 

    for od_backbone in ['fasterrcnn','ssd']:
    #for od_backbone in ['ssd']:
        print('[*] starting subexperiment for backbone type',od_backbone)
        config['object_detection_backbone'] = od_backbone
        config['object_detection_backbonepath'] = {
            'ssd': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
            'fasterrcnn': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'
        }[config['object_detection_backbone']]
        config['object_detection_batch_size'] = {'ssd': 16, 'fasterrcnn': 4}[config['object_detection_backbone']]
        
        print(config,'\n')
        now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
        checkpoint_directory = os.path.expanduser("~/checkpoints/experiments/%s/F/%s-%s" % (config['project_name'],od_backbone,now))
        finetune(config, checkpoint_directory)

        clear_session()
        print(10*'\n')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--train_video_ids',required=True)
    parser.add_argument('--test_video_ids',required=True)
    args = parser.parse_args()
    experiment_f(args)


