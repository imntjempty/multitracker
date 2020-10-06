"""
    plot results of experiments
"""

import os 
from glob import glob 
import numpy as np 
import matplotlib.pyplot as plt 
from multitracker.keypoint_detection import model 

def plot_experiment_a(args):
    print('plot',args)
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
    fig, axs = plt.subplots(1, 1)
    ltrains, ltests = [], []
    for perc_used in [1, 10,50,100]:
        if perc_used in experiment_dirs and os.path.isfile(os.path.join(experiment_dirs[perc_used],'train_log.csv')):
            # open train and test csv
            with open(os.path.join(experiment_dirs[perc_used],'train_log.csv'),'r') as f:
                train_data = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1])] for l in f.readlines()]
            with open(os.path.join(experiment_dirs[perc_used],'test_log.csv'),'r') as f:
                test_data = [[int(l.replace('\n','').split(',')[0]),float(l.replace('\n','').split(',')[1])] for l in f.readlines()]
                
            ltrains.append(axs.plot([c[0] for c in train_data],[c[1] for c in train_data],color=colors[perc_used],linestyle='--',label='train {0}% used'.format(perc_used)))
            ltests.append(axs.plot([c[0] for c in test_data],[c[1] for c in test_data],color=colors[perc_used],linestyle='-',label='test  {0}% used'.format(perc_used)))
    axs.set_title('Experiment A - using fractions of training data ({0} samples total)'.format(num_train_samples))
    axs.set_xlabel('steps')
    axs.set_ylabel('focal loss')
    
    axs.legend()
    axs.set_ylim([0.0,0.05])
    plt.yscale(["linear", "log", "symlog", "logit"][0])
    axs.grid(True)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--a')
    args = parser.parse_args()
    plot_experiment_a(args)