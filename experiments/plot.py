"""
    plot results of experiments
"""

import os 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
from multitracker.keypoint_detection import model 
from multitracker.experiments import bg_accuracy

def plot_experiment_a(args, plot=True):
    print('plot',args)
    percs = [1, 10,50,100]
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/A')
    config = model.get_config(project_id=7)
    num_train_samples = len(glob(os.path.join(config['roi_dir'],'train','*.png')))

    experiment_dirs = { }
    for checkpoint_dir in glob(base_dir+'/*/'):
        print(checkpoint_dir)
        perc_used = int(checkpoint_dir.split('/')[-2].split('-')[0])
        if not perc_used in experiment_dirs:
            experiment_dirs[perc_used] = checkpoint_dir

    colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red'}
    fig, axs = plt.subplots(2)
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
    axs[0].set_title('Experiment A - using fractions of training data ({0} samples total)'.format(num_train_samples))
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
                acc_trains.append(axs[1].plot([c[0] for c in train_data],[c[2] for c in train_data],color=colors[perc_used],linestyle='--',label='train {0}% used'.format(perc_used)))
                acc_tests.append(axs[1].plot([c[0] for c in test_data],[c[3] for c in test_data],color=colors[perc_used],linestyle='-',label='test  {0}% used'.format(perc_used)))
            except Exception as e:
                print(e)
                print()  
    axs[1].set_title('Experiment A - using fractions of training data ({0} samples total)'.format(num_train_samples))
    axs[1].set_xlabel('steps')
    axs[1].set_ylabel('pixel accuracy')
    
    axs[1].legend()
    axs[1].hlines(bg_accuracy.mice_bg_accuracy, 0, config['max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[1].set_ylim([bg_accuracy.mice_bg_accuracy-0.01,1.0])


    plt.yscale(["linear", "log", "symlog", "logit"][0])

    fig.tight_layout()
    plt.show()
    return train_dataset, test_dataset

def plot_experiment_c(args):
    num_test_samples = len(glob(os.path.join(config['roi_dir'],'test','*.png')))
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/C')
    
    # load durations json 
    with open(args.results, 'w') as f:
        durations = json.load(f)
        #for backbone in durations.keys():
        #    for bs in durations[backbone].keys():

    ## plot losses
    colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red'}
    fig, axs = plt.subplots(1)
    axs[0].set_title('Experiment C - inference loss using test data ({0} samples total)'.format(num_test_samples))
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('focal loss')
    axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[0].legend()
    axs[0].set_ylim([0.0,0.01+bg_accuracy.mice_bg_focal_loss])
    axs[0].grid(True)
    #mice_bg_cce_loss = 0.4167337
    
    # load 100 version of experiment A as largeefficientnet
    train_dataset, test_dataset = plot_experiment_a(args, plot=False)
    axs[0].plot([c[0] for c in train_dataset[100]],[c[1] for c in train_data],color=colors[100],linestyle='--',label='train EfficientNetB6')
    axs[0].plot([c[0] for c in test_dataset[100]],[c[1] for c in test_data],color=colors[100],linestyle='-',label='test  EfficientNetB6')

    # load vgg16 network
    
    #config = model.get_config(project_id=7)
    experiment_dirs = { }
    for checkpoint_dir in glob(base_dir+'/*/'):
        print(checkpoint_dir)
        backbone = int(checkpoint_dir.split('/')[-2].split('-')[0])
        if backbone == "vgg16":
            with open(os.path.join(experiment_dirs[perc_used],'train_log.csv'),'r') as f:
                train_vgg16 = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1]),float(l.replace('\n','').split(',')[2])] for l in f.readlines()]
            with open(os.path.join(experiment_dirs[perc_used],'test_log.csv'),'r') as f:
                test_vgg16 = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1]),float(l.replace('\n','').split(',')[2]),float(l.replace('\n','').split(',')[3])] for l in f.readlines()]
            axs[0].plot([c[0] for c in train_vgg16,[c[1] for c in train_data],color=colors[50],linestyle='--',label='train VGG16')
            axs[0].plot([c[0] for c in test_vgg16],[c[1] for c in test_data],color=colors[50],linestyle='-',label='test  VGG16')
        
        



if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--results')
    args = parser.parse_args()
    plot_experiment_a(args)