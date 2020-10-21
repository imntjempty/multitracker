
import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm
from multitracker.object_detection.finetune import finetune
from datetime import datetime

def experiment_f(args):
    print('[*] starting experiment F: object detection test loss SSD vs FasterRCNN')
    config = model.get_config(args.project_id)
    config['video_id'] = int(args.video_id)
    config['experiment'] = 'F'
    config['maxsteps_objectdetection'] = 50000

    #for od_backbone in ['fasterrcnn','ssd']:
    for od_backbone in ['fasterrcnn']:
        print('[*] starting subexperiment for backbone type',od_backbone)
        config['object_detection_backbone'] = od_backbone
        config['object_detection_backbonepath'] = {
            'ssd': 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8',
            'fasterrcnn': 'faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8'
        }[config['object_detection_backbone']]
        config['object_detection_batch_size'] = {'ssd': 16, 'fasterrcnn': 4}[config['object_detection_backbone']]
        
        print(config,'\n')
        now = str(datetime.now()).replace(' ','_').replace(':','-').split('.')[0]
        checkpoint_directory = os.path.expanduser("~/experiments/%s/F/%s-%s" % (config['project_name'],od_backbone,now))
        finetune(config, checkpoint_directory)

        clear_session()
        print(10*'\n')
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    experiment_f(args)


