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

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

config = model.get_config(project_id=7)
video_id = 9

config['max_steps'] = 50000
dpi=300
figsize = (12,8)
output_dir = os.path.expanduser('~/Documents/Multitracker_experiments')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

def plot_experiment_a(args, plot=True):
    print('plot',args)
    percs = [1, 10,50,100]
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/A')
    num_train_samples = len(glob(os.path.join(config['roi_dir'],'train','*.png')))

    experiment_dirs = { }
    for checkpoint_dir in glob(base_dir+'/*/'):
        print(checkpoint_dir)
        perc_used = int(checkpoint_dir.split('/')[-2].split('-')[0])
        if not perc_used in experiment_dirs:
            experiment_dirs[perc_used] = checkpoint_dir

    colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red'}
    fig, axs = plt.subplots(2)
    fig.set_size_inches(figsize[0],figsize[1])
    ltrains, ltests = [], []
    acc_trains, acc_tests = [], []
    train_dataset = {}
    test_dataset = {}
    for perc_used in percs:
        if perc_used in experiment_dirs and os.path.isfile(os.path.join(experiment_dirs[perc_used],'train_log.csv')):
            try:
                # open train and test csv
                with open(os.path.join(experiment_dirs[perc_used],'train_log.csv'),'r') as f:
                    train_data = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1]),float(l.replace('\n','').split(',')[2])] for l in f.readlines()]
                with open(os.path.join(experiment_dirs[perc_used],'test_log.csv'),'r') as f:
                    test_data = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1]),float(l.replace('\n','').split(',')[2]),float(l.replace('\n','').split(',')[3])] for l in f.readlines()]
                train_dataset[perc_used] = train_data
                test_dataset[perc_used] = test_data

                ltrains.append(axs[0].plot([c[0] for c in train_data],[c[1] for c in train_data],color=colors[perc_used],linestyle='--',label='train {0}% used'.format(perc_used)))
                ltests.append(axs[0].plot([c[0] for c in test_data],[c[1] for c in test_data],color=colors[perc_used],linestyle='-',label='test  {0}% used'.format(perc_used)))
            except Exception as e:
                print(e)
                print()    
    axs[0].set_title('Experiment A - Keypoint Estimation: using fractions of training data ({0} samples total)'.format(num_train_samples))
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('focal loss')
    axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    #mice_bg_cce_loss = 0.4167337
    
    
    axs[0].legend()
    axs[0].set_ylim([0.0,0.01+bg_accuracy.mice_bg_focal_loss])
    axs[0].grid(True)

    for perc_used in percs:
        if perc_used in experiment_dirs and os.path.isfile(os.path.join(experiment_dirs[perc_used],'train_log.csv')):
            try:
                acc_trains.append(axs[1].plot([c[0] for c in train_dataset[perc_used]],[c[2] for c in train_dataset[perc_used]],color=colors[perc_used],linestyle='--',label='train {0}% used'.format(perc_used)))
                acc_tests.append(axs[1].plot([c[0] for c in test_dataset[perc_used]],[c[3] for c in test_dataset[perc_used]],color=colors[perc_used],linestyle='-',label='test  {0}% used'.format(perc_used)))
            except Exception as e:
                print(e)
                print()  
    axs[1].set_title('Experiment A - Keypoint Estimation: using fractions of training data ({0} samples total)'.format(num_train_samples))
    axs[1].set_xlabel('steps')
    axs[1].set_ylabel('pixel accuracy')
    
    axs[1].hlines(bg_accuracy.mice_bg_accuracy, 0, config['max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[1].set_ylim([bg_accuracy.mice_bg_accuracy-0.01,1.0])
    axs[1].legend()

    plt.yscale(["linear", "log", "symlog", "logit"][0])

    fig.tight_layout()
    #plt.show()
    if plt:
        plt.savefig(os.path.join(output_dir,'A.png'), dpi=dpi)
    reset_figures()
    return train_dataset, test_dataset

def plot_experiment_b(args):
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/B')
    
    
    ## plot losses
    colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red'}
    fig, axs = plt.subplots(2)
    fig.set_size_inches(figsize[0],figsize[1])
    axs[0].set_title('Experiment B - Keypoint Estimation: ImageNet pretrained backbone vs random initialised network')
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('focal loss')
    axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    
    axs[0].set_ylim([0.0,0.01+bg_accuracy.mice_bg_focal_loss])
    axs[0].grid(True)
    #mice_bg_cce_loss = 0.4167337
    
    # load 100 version of experiment A as largeefficientnet
    train_dataset, test_dataset = plot_experiment_a(args, plot=False)
    axs[0].plot([c[0] for c in train_dataset[100]],[c[1] for c in train_dataset[100]],color=colors[100],linestyle='--',label='train fixed pretrained backbone')
    axs[0].plot([c[0] for c in test_dataset[100]],[c[1] for c in test_dataset[100]],color=colors[100],linestyle='-',label='test  fixed pretrained backbone')

    # load vgg16 network
    experiment_dirs = { }
    for checkpoint_dir in glob(base_dir+'/*/'):
        print(checkpoint_dir)
        initmethod = checkpoint_dir.split('/')[-2].split('-')[0]
        if initmethod == "random":
            with open(os.path.join(checkpoint_dir,'train_log.csv'),'r') as f:
                train_random = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1]),float(l.replace('\n','').split(',')[2])] for l in f.readlines()]
            with open(os.path.join(checkpoint_dir,'test_log.csv'),'r') as f:
                test_random = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1]),float(l.replace('\n','').split(',')[2]),float(l.replace('\n','').split(',')[3])] for l in f.readlines()]
            axs[0].plot([c[0] for c in train_random],[c[1] for c in train_random],color=colors[50],linestyle='--',label='train randomly initialised backbone')
            axs[0].plot([c[0] for c in test_random],[c[1] for c in test_random],color=colors[50],linestyle='-',label='test  randomly initialised backbone')
    
    axs[0].legend()

    ## accuracy
    axs[1].set_title('Experiment B - Keypoint Estimation: ImageNet pretrained backbone vs random initialised network'    )
    axs[1].set_xlabel('steps')
    axs[1].set_ylabel('pixel accuracy')
    axs[1].plot([c[0] for c in train_dataset[100]],[c[2] for c in train_dataset[100]],color=colors[100],linestyle='--',label='train fixed pretrained backbone')
    axs[1].plot([c[0] for c in test_dataset[100]],[c[3] for c in test_dataset[100]],color=colors[100],linestyle='-',label='test  fixed pretrained backbone')

    axs[1].plot([c[0] for c in train_random],[c[2] for c in train_random],color=colors[50],linestyle='--',label='train randomly initialised backbone')
    axs[1].plot([c[0] for c in test_random],[c[3] for c in test_random],color=colors[50],linestyle='-',label='test  randomly initialised backbone')

    axs[1].hlines(bg_accuracy.mice_bg_accuracy, 0, config['max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[1].set_ylim([bg_accuracy.mice_bg_accuracy-0.01,1.0])
    axs[1].legend()


    fig.tight_layout()
    plt.savefig(os.path.join(output_dir,'B.png'), dpi=dpi)

def plot_experiment_c(args):
    num_test_samples = len(glob(os.path.join(config['roi_dir'],'test','*.png')))
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/C')
    
    ## plot losses
    colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red'}
    fig, axs = plt.subplots(2)
    fig.set_size_inches(figsize[0],figsize[1])
    axs[0].set_title('Experiment C - Keypoint Estimation: inference loss using test data ({0} samples total)'.format(num_test_samples))
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('focal loss')
    axs[0].set_ylim([0.0,0.01+bg_accuracy.mice_bg_focal_loss])
    axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[0].grid(True)
    #mice_bg_cce_loss = 0.4167337
    
    # load 100 version of experiment A as largeefficientnet
    train_dataset, test_dataset = plot_experiment_a(args, plot=False)
    axs[0].plot([c[0] for c in train_dataset[100]],[c[1] for c in train_dataset[100]],color=colors[100],linestyle='--',label='train EfficientNetB6')
    axs[0].plot([c[0] for c in test_dataset[100]],[c[1] for c in test_dataset[100]],color=colors[100],linestyle='-',label='test  EfficientNetB6')

    # load vgg16 network
    experiment_dirs = { }
    for checkpoint_dir in glob(base_dir+'/*/'):
        print(checkpoint_dir)
        backbone = checkpoint_dir.split('/')[-2].split('-')[0]
        if backbone == "vgg16":
            with open(os.path.join(checkpoint_dir,'train_log.csv'),'r') as f:
                train_vgg16 = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1]),float(l.replace('\n','').split(',')[2])] for l in f.readlines()]
            with open(os.path.join(checkpoint_dir,'test_log.csv'),'r') as f:
                test_vgg16 = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1]),float(l.replace('\n','').split(',')[2]),float(l.replace('\n','').split(',')[3])] for l in f.readlines()]
            axs[0].plot([c[0] for c in train_vgg16],[c[1] for c in train_vgg16],color=colors[50],linestyle='--',label='train VGG16')
            axs[0].plot([c[0] for c in test_vgg16],[c[1] for c in test_vgg16],color=colors[50],linestyle='-',label='test  VGG16')
        
            # load durations json 
            with open(os.path.join(checkpoint_dir,'experiment_c_speed.json'), 'r') as f:
                durations = json.load(f)
    axs[0].legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir,'C_loss.png'), dpi=dpi)

def plot_experiment_e(args):
    #num_test_samples = len(glob(os.path.join(config['roi_dir'],'test','*.png')))
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/E')
    num_train_samples = len(db.get_labeled_bbox_frames(video_id))
    ## plot losses
    colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red'}
    fig, axs = plt.subplots(1)
    axs = [axs]
    fig.set_size_inches(figsize[0],figsize[1])
    axs[0].set_title('Experiment E - object detection: using fractions of training data ({0} samples total)'.format(num_train_samples))
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('loss')
    axs[0].set_ylim([0.0,2.])
    #axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[0].grid(True)
    
    for perc_used in [1,10,50,100]:
        exp_dir = glob(base_dir+'/%i-*/'%perc_used)[0]
        with open(os.path.join(exp_dir,'train_log.csv'),'r') as f:
            train_data = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1])] for l in f.readlines()]
        with open(os.path.join(exp_dir,'test_log.csv'),'r') as f:
            test_data = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1])] for l in f.readlines()]
        
        axs[0].plot([c[0] for c in train_data],[c[1] for c in train_data],color=colors[perc_used],linestyle='--',label='train {0}% used'.format(perc_used))
        axs[0].plot([c[0] for c in test_data],[c[1] for c in test_data],color=colors[perc_used],linestyle='-',label='test  {0}% used'.format(perc_used))
    
    axs[0].legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir,'E.png'), dpi=dpi)

def reset_figures():
    try:
        plt.close() 
    except Exception as e:
        print('reset:',e)
        try:
            plt.clf() 
        except Exception as e:
            print('reset:',e)

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    #parser.add_argument('--durationsC')
    args = parser.parse_args()
    if 0:
        plot_experiment_a(args)
        reset_figures()
    if 0:
        plot_experiment_b(args)
        reset_figures()
    if 0:
        plot_experiment_c(args)
        reset_figures()
    if 1:
        plot_experiment_e(args)
        reset_figures()
    
    print('[*] wrote plots to %s' % output_dir)