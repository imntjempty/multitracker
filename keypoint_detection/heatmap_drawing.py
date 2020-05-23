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

def draw_heatmap(image_shape, keypoint_names, keypoints, color = None, max_val = 1.0, dst_dtype = 'float32'):
    heatmap = np.zeros(image_shape,dst_dtype)

    #if color is None:
    #    colors = 

    for kp in keypoints:
        pos = (int(kp[2]),int(kp[3]))
        radius = int(0.01*np.max(heatmap.shape[:2]))

        #print('heatmap',heatmap.shape,'color',color)
        heatmap = cv.circle(heatmap, pos, radius, color,-1)
        heatmap = cv.circle(heatmap, pos, radius, (180,180,180),2)
    
    heatmap = heatmap.astype(dst_dtype)
    return heatmap 

def vis_heatmap(image, keypoint_names, keypoints):
    hsv_color = np.ones((5,5,3))*np.array([int(255.*np.random.random()),128,128])#.reshape((1,1,3))
    hsv_color = hsv_color.astype(np.uint8)
    rgb_color = cv.cvtColor(hsv_color,cv.COLOR_HSV2RGB)[0,0,:]
    #print('rgb in',image.shape)
    
    hm = draw_heatmap(image.shape, keypoint_names, keypoints, tuple([int(x) for x in rgb_color]),dst_dtype='uint8')
    vis = np.uint8( hm//2 + image//2 ) 
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
        q = "select keypoint_name, individual_id, keypoint_x, keypoint_y from keypoint_positions where video_id=%i and frame_idx='%s' order by individual_id, keypoint_name desc;" % (video_id,frame_idxs[i])
        db.execute(q)
        keypoints = [x for x in db.cur.fetchall()]
        print('frame',frame_idxs[i],'isfile',os.path.isfile(filepath),filepath)
        #for kp in keypoints:
        #    print(kp)
        if os.path.isfile(filepath):
            vis = vis_heatmap(cv.imread(filepath), keypoint_names, keypoints)
            vis_path = os.path.join(dst_dir,'%s.png' % frame_idxs[i] )
            cv.imwrite(vis_path, vis)

if __name__ == '__main__':
    project_id = 2
    randomly_drop_visualiztions(project_id)