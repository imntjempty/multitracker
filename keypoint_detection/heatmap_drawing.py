"""
    here we draw heatmaps for keypoints

    we use this for inputs as conv nets or draw nice visualizations with it
"""

import numpy as np 
import cv2 as cv 
#import tensorflow as tf 
import os 
from random import shuffle 
from multitracker.be import dbconnection

def draw_heatmap(image_shape, keypoint_names, keypoints, max_val = 1.0, dst_dtype = 'float32'):
    heatmap = np.zeros(image_shape,dst_dtype)

    
    return heatmap 

def vis_heatmap(image, keypoint_names, keypoints):
    num_idv = len(keypoints) // len(keypoint_names)
    vis = np.zeros(image.shape,'uint8')
    for i in range(num_idv):
        hm = draw_heatmap(image.shape, keypoint_names, keypoints)

    return vis 

def randomly_drop_visualiztions(project_id, num = 16):
    # take random frames from the db and show their labeling as gaussian heatmaps
    dst_dir = '/tmp/keypoint_heatmap_vis'
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    db = dbconnection.DatabaseConnection()

    keypoint_names = db.get_keypoint_names(project_id)
    video_id = db.get_random_project_video(project_id)
    if video_id is None:
        raise Exception("[ERROR] no video found for project!")

    # first get all frames 
    q = "select frame_idx from keypoint_positions;"
    db.execute(q)
    frame_idxs = [x[0] for x in db.cur.fetchall()]
    
    shuffle(frame_idxs)
    frame_idxs = frame_idxs[:min(num,len(frame_idxs))]

    for i in range(len(frame_idxs)):
        filepath = os.path.expanduser("~/data/multitracker/projects/%i/%i/frames/train/%s.png" % (int(project_id), int(video_id), frame_idxs[i]))
        q = "select keypoint_name, individual_id, keypoint_x, keypoint_y from keypoint_positions where frame_idx='%s' order by individual_id, keypoint_name desc;" % frame_idxs[i]
        db.execute(q)
        keypoints = [x for x in db.cur.fetchall()]
        print('frame',frame_idxs[i])
        for kp in keypoints:
            print(kp)
        
        vis = vis_heatmap(cv.imread(filepath), keypoint_names, keypoints)
        vis_path = os.path.join(dst_dir,'%s.png' % frame_idxs[i] )
        cv.imwrite(vis_path, vis)

if __name__ == '__main__':
    project_id = 1
    randomly_drop_visualiztions(project_id)