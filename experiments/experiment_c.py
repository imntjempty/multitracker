import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_c():
    print('[*] starting experiment C: keypoint estimation test loss/inference speed: EfficientNet vs VGG16')
    config = model.get_config(args.project_id)
    model.create_train_dataset(config)
    config['video_id'] = int(args.video_id)

    for backbone in ['efficientnetLarge','vgg16']:
        print('[*] starting sub experiment backbone %s' % backbone)
        config['experiment'] = 'C'
        config['backbone'] = backbone
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