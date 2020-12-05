import os 
import numpy as np 
import cv2 as cv 
import tensorflow as tf 
import matplotlib.pyplot as plt
import json 
import time 

from multitracker.tracking.inference import get_heatmaps_keypoints
from multitracker.keypoint_detection import roi_segm, model
from multitracker.object_detection import finetune

def calc_iou(boxA,boxB):
    ## box encoded y_min, x_min, y_max, x_max
    
    # compute the area of intersection rectangle
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def objectdetection_draw_predicision_recall_curves(video_id, title, experiment_dirs, experiment_names, output_file, mode='test'):
    num_classes, label_id_offset = 1,1
    thresh_detections = np.arange(0.1,1.,step=.05)

    ## calculate iou - source https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    assert len(experiment_dirs) > 0
    assert len(experiment_dirs) == len(experiment_names)

    figsize = (12,8)
    colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red',200: 'tab:black',300: 'tab:yellow', 400: 'tab:gray'}
    fig, axs = plt.subplots(1)
    fig.set_size_inches(figsize[0],figsize[1])
    axs.set_title(title)
    axs.set_xlabel('Recall')
    axs.set_ylabel('Precision')
    #axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['kp_max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    
    axs.set_ylim([.0,1.])
    axs.grid(True)

    iou_threshold = 0.5
    ts = time.time()
    for i, experiment_dir in enumerate(experiment_dirs):
        print('[* %i/%i] inferencing experiment' %(i,len(experiment_dirs)), experiment_names[i],'in dir',experiment_dir)
        # load config
        with open(os.path.join(experiment_dir,'config.json')) as json_file:
            config = json.load(json_file)
        config['objectdetection_model'] = experiment_dir
        #config['object_detection_batch_size'] = 2
        
        frame_bboxes, data_train, data_test = finetune.get_bbox_data(config, video_ids=str(video_id))
        data = data_train
        if mode == 'test':
            data = data_test 
        gt_boxes, gt_classes = [], []
        detection_model = finetune.load_trained_model(config)

        precisions, recalls = {},{}
        cnt_true_positives, cnt_false_positives, cnt_false_negatives = {},{},{}
        for thresh_detection in thresh_detections:
            cnt_true_positives[thresh_detection], cnt_false_positives[thresh_detection], cnt_false_negatives[thresh_detection] = 0,0,0

        for frame_idx, image_tensors in data:
            gt_boxes, gt_classes = [],[]
            for ii in range(len(frame_idx)):
                gt_boxes.append(tf.convert_to_tensor(frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )], dtype=tf.float32))
                gt_classes.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )].shape[0]], dtype=np.int32) - label_id_offset), num_classes))
        
            preprocessed_images, shapes = detection_model.preprocess(image_tensors) 
            prediction_dict = detection_model.predict(preprocessed_images, shapes)
            prediction_dict = detection_model.postprocess(prediction_dict, shapes)
            
            for thresh_detection in thresh_detections:
                for b in range(len(frame_idx)):
                    detections = [ prediction_dict['detection_boxes'][b][j] for j in range(len(prediction_dict['detection_boxes'][b])) if prediction_dict['detection_scores'][b][j] > thresh_detection]
                    
                    for j in range(len(detections)):
                        det_matched = False
                        for k in range(len(gt_boxes[b])):
                            iou = calc_iou(detections[j],gt_boxes[b][k])
                            if iou > iou_threshold:
                                det_matched = True 
                        if det_matched:
                            cnt_true_positives[thresh_detection]+=1
                        else:
                            cnt_false_positives[thresh_detection]+=1 

                    for k in range(len(gt_boxes[b])):
                        gt_matched = False
                        for j in range(len(detections)):
                            iou = calc_iou(detections[j],gt_boxes[b][k])
                            if iou > iou_threshold:
                                gt_matched = True 
                        if not gt_matched:
                            cnt_false_negatives[thresh_detection]+=1 


        # calculate precision and recall for this experiment
        for thresh_detection in thresh_detections:
            precisions[thresh_detection] = cnt_true_positives[thresh_detection] / max(1e-5,cnt_true_positives[thresh_detection] + cnt_false_positives[thresh_detection])
            recalls[thresh_detection] = cnt_true_positives[thresh_detection] / max(1e-5,cnt_true_positives[thresh_detection] + cnt_false_negatives[thresh_detection])
            print('exp',experiment_names[i],thresh_detection,'->',precisions[thresh_detection],recalls[thresh_detection])
                
        axs.plot([recalls[th] for th in thresh_detections],[precisions[th] for th in thresh_detections],color=colors[[1,10,50,100,200,300,400][i%7]],linestyle='-',label=experiment_names[i])
        #axs[0].plot([c[0] for c in test_random],[c[1] for c in test_random],color=colors[50],linestyle='-',label='test  randomly initialised backbone')

    print('[*] %s time took %f seconds' % (output_file,time.time()-ts))
    axs.legend()
    fig.tight_layout()
    plt.savefig(output_file, dpi=300)
    return precisions, recalls, output_file


def keypoints_draw_predicision_recall_curves(video_id, title, experiment_dirs, experiment_names, output_file, max_neighbor_dist = 10, mode = 'test'):
    """
        draw a graph that compares precision and recall curves for thresholds in [0.1,0.2,...,0.9] of different keypoint prediction experiments

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

        for each experiment:
            a) inference test set
            b) extract keypoints for varying thresholds
            c) calculate minimum distance for each detection box to all ground truth boxes
            d) count match == if really close by
        e) draw graph
    """
    assert len(experiment_dirs) > 0
    assert len(experiment_dirs) == len(experiment_names)

    ## plot curves
    figsize = (12,8)
    colors = {1: 'tab:brown', 10: 'tab:blue',20: 'tab:orange', 50: 'tab:green', 100: 'tab:red'}
    fig, axs = plt.subplots(1)
    fig.set_size_inches(figsize[0],figsize[1])
    axs.set_title(title)
    axs.set_xlabel('Recall')
    axs.set_ylabel('Precision')
    #axs[0].hlines(bg_accuracy.mice_bg_focal_loss, 0, config['kp_max_steps'], colors='k', linestyles='solid', label='baseline - no keypoints')
    
    axs.set_ylim([.5,1.])
    axs.grid(True)


    thresh_detections = np.arange(0.1,1.,step=.1)
    for i, experiment_dir in enumerate(experiment_dirs):
        ts = time.time()
        print('[* %i/%i] inferencing experiment' %(i,len(experiment_dirs)), experiment_names[i],'in dir',experiment_dir)
        # load config
        with open(os.path.join(experiment_dir,'config.json')) as json_file:
            config = json.load(json_file)
        config['batch_size'] = 128
        print(config)
        
        dataset = roi_segm.load_roi_dataset(config,mode=mode,video_ids=str(video_id))

        # load net 
        net = model.get_model(config) # outputs: keypoints + background
        ckpt = tf.train.Checkpoint(net = net)
        ckpt_manager = tf.train.CheckpointManager(ckpt, experiment_dir, max_to_keep=5)

        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

        # a) inference
        recalls, precisions, count_matches = {}, {}, {}
        cnt_true_positives, cnt_false_negatives, cnt_false_positives = {}, {}, {}
        for thresh_detection in thresh_detections:
            precisions[thresh_detection], recalls[thresh_detection], count_matches[thresh_detection] = 0,0,0
            cnt_true_positives[thresh_detection] = 0 
            cnt_false_negatives[thresh_detection] = 0
            cnt_false_positives[thresh_detection] = 0 
        
        cnt_batches = -1
        for xt,yt in dataset:
            cnt_batches += 1 
            predicted_test = net(xt,training=False)[-1].numpy()
            for b in range(0,predicted_test.shape[0],1):
                keypoints_gt = get_heatmaps_keypoints(yt.numpy()[b,:,:,:],thresh_detection=0.8)

                for thresh_detection in thresh_detections:
                    keypoints = get_heatmaps_keypoints(predicted_test[b,:,:,:], thresh_detection=thresh_detection)
                    # nearest neighbour to all ground truth points
                    for jgt, kp_gt in enumerate(keypoints_gt):
                        gt_kp_matched = False
                        for j, kp in enumerate(keypoints):
                            # match iff small distance and same class
                            _dist = np.linalg.norm(np.array(kp[:2])-np.array(kp_gt[:2]))
                            if _dist < max_neighbor_dist and kp[2]==kp_gt[2]:
                                count_matches[thresh_detection] += 1
                                gt_kp_matched = True 
                        if not gt_kp_matched:
                            cnt_false_negatives[thresh_detection] += 1 # unmatched ground truth

                    for j, kp in enumerate(keypoints):
                        kp_matched = False
                        for jgt, kp_gt in enumerate(keypoints_gt):
                            # match iff small distance and same class
                            _dist = np.linalg.norm(np.array(kp[:2])-np.array(kp_gt[:2]))
                            if _dist < max_neighbor_dist and kp[2]==kp_gt[2]:
                                kp_matched = True 
                        if kp_matched: # unmatched prediction
                            cnt_true_positives[thresh_detection] += 1
                        else:
                            cnt_false_positives[thresh_detection] += 1
            
        # calculate precision and recall for this experiment
        for thresh_detection in thresh_detections:
            precisions[thresh_detection] = cnt_true_positives[thresh_detection] / max(1e-5,cnt_true_positives[thresh_detection] + cnt_false_positives[thresh_detection])
            recalls[thresh_detection] = cnt_true_positives[thresh_detection] / max(1e-5,cnt_true_positives[thresh_detection] + cnt_false_negatives[thresh_detection])
            print('exp',experiment_names[i],thresh_detection,'->',precisions[thresh_detection],recalls[thresh_detection])
            
        axs.plot([recalls[th] for th in thresh_detections],[precisions[th] for th in thresh_detections],color=colors[[1,10,50,100,200,300,400][i%7]],linestyle='-',label=experiment_names[i])
        #axs[0].plot([c[0] for c in test_random],[c[1] for c in test_random],color=colors[50],linestyle='-',label='test  randomly initialised backbone')

        print('[*] %s time took %f seconds' % (output_file, time.time()-ts))
    axs.legend()
    fig.tight_layout()
    plt.savefig(output_file, dpi=300)

    # 1) extract keypoints
    # 2) calculate minimum distance for each detection box to all ground truth boxes
    # 3) count match == if really close by

    
    
if __name__ == "__main__":
    if 1:
        output_file = '/tmp/objectdetection_rois.png'
        #objectdetection_draw_predicision_recall_curves( 'Experiment Q - huhu', ['/home/alex/checkpoints/multitracker/bbox/vids9,14-2020-11-12_17-04-53','/home/alex/checkpoints/multitracker/bbox/vids9,14-2020-11-13_07-56-02'],['with flip','without flip'], output_file)
        objectdetection_draw_predicision_recall_curves( 13, 'Experiment Q - huhu', ['/home/alex/checkpoints/multitracker/bbox/vids9,14-2020-11-13_07-56-02'],['without flip'], output_file)
        print('[*] wrote',output_file)

    if 0:
        output_file = '/tmp/keypoint_rois.png'
        keypoints_draw_predicision_recall_curves('Experiment K - have fun', ['/home/alex/checkpoints/multitracker/keypoints/vids9,14-2020-11-10_10-09-25','/home/alex/checkpoints/multitracker/keypoints/vids9,14-2020-11-12_08-41-41'], ['less data', 'heavy bg sampling'], output_file, max_neighbor_dist = 10)
        print('[*] wrote',output_file)