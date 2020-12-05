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

config = model.get_config(project_id=7)
video_id = 9

config['max_steps'] = 25000
dpi=300
figsize = (12,8)
output_dir = os.path.expanduser('~/Documents/Multitracker_experiments')
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red',200: 'tab:black',300: 'tab:yellow', 400: 'tab:gray'}

def plot_experiment_a_roi(args):
    num_train_samples = len(db.get_labeled_bbox_frames(args.video_id))
    title = 'Experiment A - Keypoint Estimation: using fractions of training data ({0} samples total)'.format(num_train_samples)
    experiment_dirs = [
        '/home/alex/checkpoints/experiments/MiceTop/A/1-2020-11-14_11-30-28',
        '/home/alex/checkpoints/experiments/MiceTop/A/10-2020-11-14_14-16-35',
        '/home/alex/checkpoints/experiments/MiceTop/A/50-2020-11-14_17-08-45',
        '/home/alex/checkpoints/experiments/MiceTop/A/100-2020-11-14_20-04-06'
    ]
    experiment_names = ['1%','10%','50%','100%']
    output_file = os.path.join(output_dir,'A_prec_recall_curve.png')
    roi_curve.keypoints_draw_predicision_recall_curves(str(args.video_id), title, experiment_dirs, experiment_names, output_file)

def plot_experiment_b_roi(args):
    num_train_samples = len(db.get_labeled_bbox_frames(args.video_id))
    title = 'Experiment B - Keypoint Estimation: ImageNet pretrained backbone vs random initialised network'
    experiment_dirs = [
        '/home/alex/checkpoints/experiments/MiceTop/A/100-2020-11-14_20-04-06',
        '/home/alex/checkpoints/experiments/MiceTop/B/random-2020-11-14_23-01-54'
    ]
    experiment_names = ['pretrained','random init']
    output_file = os.path.join(output_dir,'B_prec_recall_curve.png')
    roi_curve.keypoints_draw_predicision_recall_curves(str(args.video_id), title, experiment_dirs, experiment_names, output_file)

def plot_experiment_c_roi(args):
    num_train_samples = len(db.get_labeled_bbox_frames(args.video_id))
    title = 'Experiment C - Keypoint Estimation: different architecture backbones'
    experiment_dirs = [
        '/home/alex/checkpoints/experiments/MiceTop/C/vgg16-2020-11-15_12-33-53',
        '/home/alex/checkpoints/experiments/MiceTop/C/efficientnetLarge-2020-11-15_14-32-57',
        '/home/alex/checkpoints/experiments/MiceTop/A/100-2020-11-14_20-04-06',
        '/home/alex/checkpoints/experiments/MiceTop/C/hourglass-4-efficientnetLarge-2020-11-15_19-16-22',
        '/home/alex/checkpoints/experiments/MiceTop/C/hourglass-8-efficientnetLarge-2020-11-15_23-57-27',
        '/home/alex/checkpoints/experiments/MiceTop/C/psp-2020-11-15_17-12-42'
    ]
    experiment_names = ['U-Net VGG16', 'U-Net Efficientnet','Stacked Hourglass 2','Stacked Hourglass 4','Stacked Hourglass 8','PSP']
    output_file = os.path.join(output_dir,'C_prec_recall_curve.png')
    roi_curve.keypoints_draw_predicision_recall_curves(str(args.video_id), title, experiment_dirs, experiment_names, output_file)

def plot_object_detector_train_traintest_testtest(args):
    title = 'Experiment G - Object Detection: generalization'

    raise Exception('''TODO!
        make precision recall curves
        train data - test data of train video - test on unseen video
    ''')
def plot_keypoint_detector_train_traintest_testtest(args):
    title = 'Experiment H - Keypoint Estimation: generalization'

    raise Exception('''TODO!
        make precision recall curves
        train data - test data of train video - test on unseen video
    ''')
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
    axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['kp_max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
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
    
    axs[1].hlines(bg_accuracy.mice_bg_accuracy, 0, config['kp_max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[1].set_ylim([bg_accuracy.mice_bg_accuracy-0.01,1.0])
    axs[1].legend()

    plt.yscale(["linear", "log", "symlog", "logit"][0])

    fig.tight_layout()
    #plt.show()
    if plt:
        plt.savefig(os.path.join(output_dir,'A_loss.png'), dpi=dpi)
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
    axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['kp_max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    
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

    axs[1].hlines(bg_accuracy.mice_bg_accuracy, 0, config['kp_max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    axs[1].set_ylim([bg_accuracy.mice_bg_accuracy-0.01,1.0])
    axs[1].legend()


    fig.tight_layout()
    plt.savefig(os.path.join(output_dir,'B_loss.png'), dpi=dpi)

def plot_experiment_c(args):
    num_test_samples = len(glob(os.path.join(config['roi_dir'],'test','*.png')))
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/C')
    
    ## plot losses
    fig, axs = plt.subplots(2)
    fig.set_size_inches(figsize[0],figsize[1])
    axs[0].set_title('Experiment C - Keypoint Estimation: inference loss using test data ({0} samples total)'.format(num_test_samples))
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('focal loss')
    axs[0].set_ylim([0.0,0.01+bg_accuracy.mice_bg_focal_loss])
    axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['kp_max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
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

    ## plot speed
    for backbone in durations.keys():
        for bs in durations[backbone].keys():
            ''#print('kp_backbone',backbone,bs,':', durations[backbone][bs])
    labels = ['1','4','16']
    vgg_durations = [durations['vgg16'][bs] for bs in labels]
    efficient_durations = [durations['efficientnetLarge'][bs] for bs in labels]
    w = 0.35

    for bs in [1,4,16]:
        rectsvgg = axs[1].bar(np.arange(len(labels)) - w/2, vgg_durations, w, label='VGG16')
        rectsefficient = axs[1].bar(np.arange(len(labels)) + w/2, efficient_durations, w, label='EfficientNet B6')

    axs[1].set_xticks(np.arange(len(labels)))
    axs[1].set_xticklabels(labels)
    axs[1].legend()

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir,'C_loss.png'), dpi=dpi)

def plot_experiment_e(args):
    if 1:
        plot_experiment_e_roi(args)
    if 1:
        plot_experiment_e_loss(args)

def plot_experiment_e_roi(args):
    num_train_samples = len(db.get_labeled_bbox_frames(args.video_id))
    title = 'Experiment E - Faster R-CNN: using fractions of training data ({0} samples total)'.format(num_train_samples)
    experiment_dirs = [
        '/home/alex/checkpoints/experiments/MiceTop/E/1-2020-12-02_11-46-27',
        '/home/alex/checkpoints/experiments/MiceTop/E/10-2020-12-02_17-02-27',
        '/home/alex/checkpoints/experiments/MiceTop/E/50-2020-12-02_22-19-06',
        '/home/alex/checkpoints/experiments/MiceTop/E/100-2020-12-03_08-42-30'
    ]
    experiment_names = ['1%','10%','50%','100%']
    output_file = os.path.join(output_dir,'E_prec_recall_curve.png')
    roi_curve.objectdetection_draw_predicision_recall_curves(str(args.video_id), title, experiment_dirs, experiment_names, output_file)

def plot_experiment_e_loss(args):
    #num_test_samples = len(glob(os.path.join(config['roi_dir'],'test','*.png')))
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/E')
    num_train_samples = len(db.get_labeled_bbox_frames(video_id))
    ## plot losses
    fig, axs = plt.subplots(1)
    axs = [axs]
    fig.set_size_inches(figsize[0],figsize[1])
    axs[0].set_title('Experiment E - object detection: using fractions of training data ({0} samples total)'.format(num_train_samples))
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('loss')
    axs[0].set_ylim([0.0,5.5])
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
    plt.savefig(os.path.join(output_dir,'E_loss.png'), dpi=dpi)

def plot_experiment_f(args):
    if 1:
        plot_experiment_f_roi(args)
    if 0:
        plot_experiment_f_loss(args)

def plot_experiment_f_roi(args):
    num_train_samples = len(db.get_labeled_bbox_frames(args.video_id))
    title = 'Experiment F - Faster R-CNN vs SSD'
    experiment_dirs = [
        '/home/alex/checkpoints/experiments/MiceTop/E/100-2020-12-03_08-42-30',
        '/home/alex/checkpoints/experiments/MiceTop/F/ssd-2020-12-04_02-37-21'
    ]
    experiment_names = ['Faster R-CNN','SSD']
    output_file = os.path.join(output_dir,'F_prec_recall_curve.png')
    roi_curve.objectdetection_draw_predicision_recall_curves(str(args.video_id), title, experiment_dirs, experiment_names, output_file)


def plot_experiment_f_loss(args):
    #num_test_samples = len(glob(os.path.join(config['roi_dir'],'test','*.png')))
    base_dir = os.path.expanduser('~/checkpoints/experiments/MiceTop/F')
    num_train_samples = len(db.get_labeled_bbox_frames(video_id))
    ## plot losses
    fig, axs = plt.subplots(1)
    axs = [axs]
    fig.set_size_inches(figsize[0],figsize[1])
    axs[0].set_title('Experiment F - object detection: SSD vs Faster R-CNN')
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('loss')
    axs[0].set_ylim([0.0,2.])
    #axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['kp_max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
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
    plt.savefig(os.path.join(output_dir,'F_loss.png'), dpi=dpi)

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
    parser.add_argument('--project_id', type=int, required=True)
    parser.add_argument('--video_id', type=int, required=True)
    args = parser.parse_args()
    if 0:
        plot_experiment_a_roi(args)
        reset_figures()
    if 0:
        plot_experiment_a(args)
        reset_figures()
    if 0:
        plot_experiment_b_roi(args)
        reset_figures()
    if 0:
        plot_experiment_b(args)
        reset_figures()
    if 1:
        plot_experiment_c_roi(args)
        reset_figures()
    if 0:
        plot_experiment_c(args)
        reset_figures()
    if 0:
        plot_experiment_e(args)
        reset_figures()
    if 0:
        plot_experiment_f(args)
        reset_figures()
    if 1:
        plot_object_detector_train_traintest_testtest(args)
        reset_figures()
    if 1:
        plot_keypoint_detector_train_traintest_testtest(args)
        reset_figures()
    print('[*] wrote plots to %s' % output_dir)