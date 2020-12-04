"""
    calculate baseline accuracy if one just predicts background for every pixel 
"""
import os 
import numpy as np 
import tensorflow as tf 
from multitracker.keypoint_detection import model, roi_segm

mice_bg_focal_loss = 0.14292024
mice_bg_cce_loss = 0.4167337
mice_bg_accuracy = 0.98160404

def get_background_baseline_loss_accuracy(config):
    dataset_test = roi_segm.load_roi_dataset(config, mode='test')
    
    fixed_background = np.ones((config['batch_size'],config['img_height'],config['img_width'],1+len(config['keypoint_names'])))
    fixed_background[:,:,:,:len(config['keypoint_names'])] = 0.0
    fixed_background = fixed_background.astype(np.float32)

    nt = 0
    test_losses = {'focal': 0.0, 'cce': 0.0, 'l2':0.0}
    test_accuracy = 0.0
    for xt,yt in dataset_test:
        predicted_test = fixed_background
        if yt.shape[0] < predicted_test.shape[0]:
            predicted_test = predicted_test[:yt.shape[0],:,:,:]
        if not predicted_test.shape[1] == yt.shape[1]:
            predicted_test = tf.image.resize(predicted_test, x.shape[1:3]) 

        if 'focal' in config['kp_test_losses']:
            test_losses['focal'] += roi_segm.calc_focal_loss(yt,predicted_test)
        if 'cce' in config['kp_test_losses']:
            test_losses['cce'] += roi_segm.calc_cce_loss(yt,predicted_test)
        if 'l2' in config['kp_test_losses']:
            test_losses['l2'] += roi_segm.calc_l2_loss(yt,predicted_test)
        test_accuracy += roi_segm.calc_accuracy(config, yt,predicted_test)
        nt += 1 
    test_losses['focal'] = test_losses['focal'] / nt
    test_losses['cce'] = test_losses['cce'] / nt
    test_losses['l2'] = test_losses['l2'] / nt
    test_accuracy = test_accuracy / nt 
    return test_losses, test_accuracy

if __name__ == "__main__":
    config = model.get_config(project_id=7)
    config['video_id'] = 9
    config['kp_test_losses'] = ['focal','cce','l2']
    baseline_losses, baseline_accuracy = get_background_baseline_loss_accuracy(config)
    print('baseline_losses')
    for k,v in baseline_losses.items():
        print('   ',k,v)
    print('baseline_accuracy',baseline_accuracy.numpy())