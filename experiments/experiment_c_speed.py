import numpy as np 
import os 
import time 
import tensorflow as tf 
from tensorflow.keras.backend import clear_session
from multitracker.keypoint_detection import model, roi_segm

def experiment_c_speed():
    print('[*] starting experiment C: keypoint estimation inference speed: EfficientNetB6 vs VGG16')
    config = model.get_config(args.project_id)
    model.create_train_dataset(config)
    config['video_id'] = int(args.video_id)

    config['experiment'] = 'C'
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

    
    durations = {'efficientnetLarge':{},'vgg16':{}}
    for backbone in ['efficientnetLarge','vgg16']:
        checkpoint_path = {
            'efficientnetLarge':'/home/alex/checkpoints/experiments/MiceTop/A/100-2020-10-10_10-14-17',
            'vgg16':''
        }[backbone]
        print('[*] starting sub experiment backbone %s' % backbone)
        config['backbone'] = backbone
        print(config,'\n')
    
        # load pretrained network
        net = unet.get_model(config) # outputs: keypoints + background
        ckpt = tf.train.Checkpoint(net = net)
    
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

        for bs in [1,4,16]:
            dataset_test = roi_segm.load_roi_dataset(config,mode='test',batch_size=bs)
            t0 = time.time()
            for xt, yt in dataset_test:
                _ = net(xt,training=False)
            t1 = time.time()
            durations[backbone][bs] = t1-t0
            print('[*] duration',backbone,'batch size',bs,'took',durations[backbone][bs])
 
        clear_session()
    # write result as JSON
    file_json = os.path.join(checkpoint_path,'experiment_c_speed.json')
    with open(file_json, 'w') as f:
        json.dump(durations, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    experiment_c(args)