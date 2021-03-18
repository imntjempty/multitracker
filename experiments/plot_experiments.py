"""
    plot results of experiments
"""

import os 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
import json
from multitracker.keypoint_detection import model 
from multitracker.experiments import bg_accuracy
from multitracker.experiments import roi_curve
from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

project_experiment_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop')

train_ids = [9,14]
test_ids = [13,14]
test_id_in = 14
test_id_out = 13

if 0:
    train_ids = [1,3]
    test_ids = [2,3]
    test_id_in = 3
    test_id_out = 2


colors = ['tab:brown', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:yellow', 'tab:gray', 'tab:cyan', 'tab:olive']

def parse_train_csv(train_csv):
    steps, losses, accuracies = [],[],[]
    with open(train_csv,'r') as f:
        lines = f.readlines()
        lines = [l.replace('\n','') for l in lines]
        for line in lines:
            step, loss, accuracy = line.split(',')
            steps.append(int(step))
            losses.append(float(loss))
            accuracy = min(float(accuracy),1.)
            accuracies.append(accuracy)
        return steps, losses, accuracies

def parse_test_csv(test_csv):
    steps, losses_focal, losses_cce, losses_l2, accuracies = [],[],[],[],[]
    with open(test_csv,'r') as f:
        lines = f.readlines()
        lines = [l.replace('\n','') for l in lines]
        for line in lines:
            step, loss_focal, loss_cce, loss_l2, accuracy = line.split(',')
            steps.append(int(step))
            losses_focal.append(float(loss_focal))
            losses_cce.append(float(loss_cce))
            losses_l2.append(float(loss_l2))
            accuracy = min(float(accuracy),1.)
            accuracies.append(accuracy)
        return steps, losses_focal, losses_cce, losses_l2, accuracies
        
def parse_objectdetect_csv(file_csv):
    steps, losses = [], []
    with open(file_csv,'r') as f: 
        lines = f.readlines()
        lines = [l.replace('\n','') for l in lines]
        for line in lines:
            step, loss = line.split(',')
            steps.append(int(step))
            losses.append(float(loss))
        return steps, losses 
        
def reset_figures():
    try:
        plt.close() 
    except Exception as e:
        print('reset:',e)
        try:
            plt.clf() 
        except Exception as e:
            print('reset:',e)

def plot_objectdetection_experiment(checkpoint_base_dir, title, experiment_names):
    """
        make 2 figures for object detection experiment
        F1 plot losses
        F1 plot pixel accuracy
        F2 plot precision recall curve

        plot test_out solid lines, test_in dashed, train dotted
    """
    figsize = (12,12)
    dpi = 300

    experiment_dirs = glob(checkpoint_base_dir + '/*/')
    experiment_dirs = sorted(experiment_dirs)

    fig, axs = plt.subplots(2)
    fig.set_size_inches(figsize[0],figsize[1])

    axs[0].set_title(title)
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('loss')
    
    #max_steps = config['kp_max_steps']
    max_steps = 15000
    axs[0].set_ylim([0.0,1.2])
    axs[0].grid(True)
    
    for i, experiment_dir in enumerate(experiment_dirs):
        print(i,experiment_dir,experiment_names[i])
        # load train log and two test logs
        train_csv = os.path.join(experiment_dir, 'train_%s_log.csv' % ','.join([str(iid) for iid in train_ids]))
        test_in_csv = os.path.join(experiment_dir, 'test_%i_log.csv' % test_id_in)
        test_out_csv = os.path.join(experiment_dir, 'test_%i_log.csv' % test_id_out)
        train_data = parse_objectdetect_csv(train_csv)
        test_in_data = parse_objectdetect_csv(test_in_csv)
        test_out_data = parse_objectdetect_csv(test_out_csv)
        #print('train_data',train_data); print('test in',test_in_data); print('test out',test_out_data); 

        # plot losses
        #axs[0].plot(train_data[0], train_data[1], color = colors[i], linestyle='dotted', label = 'train ' + experiment_names[i])
        axs[0].plot(test_in_data[0], test_in_data[1], color = colors[i], linestyle='dashed', label = 'test seen video ' + experiment_names[i])
        axs[0].plot(test_out_data[0], test_out_data[1], color = colors[i], linestyle='solid', label = 'test unseen video ' + experiment_names[i])

    axs[0].legend()
    fig.tight_layout()
    #plt.show()
    #if plt:
    file_plot = os.path.join(project_experiment_dir,'%sloss.png' % checkpoint_base_dir.split('/')[-1])
    plt.savefig(file_plot, dpi=dpi)
    print('[*] wrote plot %s for experiment %s' % (file_plot, title))
    reset_figures()

    ## plot precision recall curve
    video_ids = [test_id_in, test_id_out]
    file_curve = os.path.join(project_experiment_dir,'%sprec.png' % checkpoint_base_dir.split('/')[-1])
    roi_curve.objectdetection_draw_predicision_recall_curves(video_ids, title, experiment_dirs, experiment_names, file_curve, mode='test')
    
    #plt.savefig(file_curve, dpi=dpi)
    print('[*] wrote plot %s for experiment %s' % (file_curve, title))
    reset_figures()

def plot_keypoint_experiment(checkpoint_base_dir, title, experiment_names):
    """
        make 2 figures for keypoint experiment
        F1 plot losses
        F1 plot pixel accuracy
        F2 plot precision recall curve

        plot test_out solid lines, test_in dashed, train dotted
    """
    figsize = (12,12)
    dpi = 300

    experiment_dirs = glob(checkpoint_base_dir + '/*/')
    experiment_dirs = sorted(experiment_dirs)

    fig, axs = plt.subplots(2)
    fig.set_size_inches(figsize[0],figsize[1])

    axs[0].set_title(title)
    axs[1].set_xlabel('steps')
    axs[0].set_ylabel('focal loss')
    axs[1].set_ylabel('pixel accuracy')
    #max_steps = config['kp_max_steps']
    max_steps = 15000
    axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, max_steps, colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[1].hlines(bg_accuracy.mice_bg_accuracy, 0, max_steps, colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[0].set_ylim([0.0,0.01+bg_accuracy.mice_bg_focal_loss])
    axs[1].set_ylim([bg_accuracy.mice_bg_accuracy-0.01,1.0])
    axs[0].grid(True)
    axs[1].grid(True)
    for i, experiment_dir in enumerate(experiment_dirs):
        print(i,experiment_dir,experiment_names[i])
        # load train log and two test logs
        train_csv = os.path.join(experiment_dir, 'train_log.csv')
        test_in_csv = os.path.join(experiment_dir, 'test_%i_log.csv' % test_id_in)
        test_out_csv = os.path.join(experiment_dir, 'test_%i_log.csv' % test_id_out)
        train_data = parse_train_csv(train_csv)
        test_in_data = parse_test_csv(test_in_csv)
        test_out_data = parse_test_csv(test_out_csv)
        #print('train_data',train_data); print('test in',test_in_data); print('test out',test_out_data); 

        # plot losses
        #axs[0].plot(train_data[0], train_data[1], color = colors[i], linestyle='dotted', label = 'train ' + experiment_names[i])
        axs[0].plot(test_in_data[0], test_in_data[1], color = colors[i], linestyle='dashed', label = 'test seen video ' + experiment_names[i])
        axs[0].plot(test_out_data[0], test_out_data[1], color = colors[i], linestyle='solid', label = 'test unseen video ' + experiment_names[i])

        # plot pixel accuracy
        #axs[1].plot(train_data[0], train_data[2], color = colors[i], linestyle='dotted', label = 'train ' + experiment_names[i])
        axs[1].plot(test_in_data[0], test_in_data[4], color = colors[i], linestyle='dashed', label = 'test seen video ' + experiment_names[i])
        axs[1].plot(test_out_data[0], test_out_data[4], color = colors[i], linestyle='solid', label = 'test unseen video ' + experiment_names[i])

    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    #plt.show()
    #if plt:
    file_plot = os.path.join(project_experiment_dir,'%sloss.png' % checkpoint_base_dir.split('/')[-1])
    plt.savefig(file_plot, dpi=dpi)
    print('[*] wrote plot %s for experiment %s' % (file_plot, title))
    reset_figures()







    ## plot precision recall curve
    video_ids = [test_id_in, test_id_out]
    file_curve = os.path.join(project_experiment_dir,'%sprec.png' % checkpoint_base_dir.split('/')[-1])
    roi_curve.keypoints_draw_predicision_recall_curves(video_ids, title, experiment_dirs, experiment_names, file_curve, max_neighbor_dist = 10, mode = 'test')

    #plt.savefig(file_curve, dpi=dpi)
    print('[*] wrote plot %s for experiment %s' % (file_curve, title))
    reset_figures()

def get_train_kp_samples(train_ids):
    num_train_samples = 0
    for train_id in train_ids:
        num_train_samples += db.get_count_labeled_frames(train_id)
    return num_train_samples
def get_train_od_samples(train_ids):
    num_train_samples = 0
    for train_id in train_ids:
        num_train_samples += db.get_count_labeled_bbox_frames(train_id)
    return num_train_samples

def plot_a():
    num_train_samples = get_train_kp_samples(train_ids)
    title = 'Experiment A - Keypoint Estimation: using fractions of training data ({0} cropped animals total)'.format(num_train_samples)
    plot_keypoint_experiment(project_experiment_dir+'/A',title,['1%','10%','100%','50%'])

def plot_b():
    num_train_samples = get_train_kp_samples(train_ids)
    title = 'Experiment B - Keypoint Estimation: pretrained encoder vs randomly initialized encoder w/ and wo/ augmentation'
    plot_keypoint_experiment(project_experiment_dir+'/B',title,['pretrained /wo augmentation','pretrained /w augmentation','random init /wo augmentation','random init /w augmentation'])

def plot_c():
    num_train_samples = get_train_kp_samples(train_ids)
    title = 'Experiment C - Keypoint Estimation: network architectures'
    plot_keypoint_experiment(project_experiment_dir+'/C',title,['Efficient-Net','Hourglass-2','Hourglass-4','Hourglass-8','Pyramid Scene Parsing','VGG16'])

def plot_d():
    title = 'Experiment D - Keypoint Estimation: loss funcions'
    plot_keypoint_experiment(project_experiment_dir+'/D',title,['cce', 'focal', 'l2'])

def plot_e():
    num_train_samples = get_train_od_samples(train_ids)
    title = 'Experiment E - Object Detection: using fractions of training data ({0} labeled frames total)'.format(num_train_samples)
    plot_objectdetection_experiment(project_experiment_dir+'/E',title,['1%','10%','100%','50%'])

def plot_f():
    num_train_samples = get_train_od_samples(train_ids)
    title = 'Experiment F - Object Detection: Faster R-CNN vs SSD'
    plot_objectdetection_experiment(project_experiment_dir+'/F',title,['Faster R-CNN','SSD'])

def plot_g():
    num_train_samples = get_train_od_samples(train_ids)
    title = 'Experiment G - Object Detection: pretrained encoder vs randomly initialized encoder w/ and wo/ augmentation'
    plot_objectdetection_experiment(project_experiment_dir+'/G',title,['pretrained augmented','randomly initialized augmented','pretrained not augmented','randomly initialized not augmented'])

if __name__ == '__main__':
    import time 
    t0 = time.time()
    #plot_d()
    #plot_a()
    #plot_b()
    #plot_c()
    #plot_e()
    #plot_f()
    plot_g()

    t1 = time.time()
    print('[*] plotting of experiments took %f minutes.' % ((t1-t0)/60.))