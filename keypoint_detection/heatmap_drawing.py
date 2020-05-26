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


def gaussian_k(x0,y0,sigma, height, width):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float) ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def generate_hm(height, width ,landmarks, keypoint_names, s=13):
    ## https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html 
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing landmarks
        maxlenght : Lenght of the Bounding Box
    """
    hm = 0.* np.ones((height, width, len(keypoint_names)), dtype = np.float32)
    for i in range(len(landmarks)):
        idx = keypoint_names.index(landmarks[i][2])
        x = gaussian_k(landmarks[i][0],
                                landmarks[i][1],
                                s,height, width)
        hm[:,:,idx] = x
        #hm[:,:,idx][x>0.1] = x 
    #hm[hm<0.3] = 128.
    return hm


def vis_heatmap(image, keypoint_names, keypoints, horistack=True):
    #hsv_color = np.ones((5,5,3))*np.array([int(255.*np.random.random()),128,128])#.reshape((1,1,3))
    #hsv_color = hsv_color.astype(np.uint8)
    #rgb_color = cv.cvtColor(hsv_color,cv.COLOR_HSV2RGB)[0,0,:]
    hm = generate_hm(image.shape[0], image.shape[1] , [ [int(kp[2]),int(kp[3]),kp[0]] for kp in keypoints ], keypoint_names)
    #print('hm',hm.shape,hm.min(),hm.max())
    if not horistack:
        # overlay
        hm = np.uint8(255. * hm[:,:,:3])
        vis = np.uint8( hm//2 + image//2 )     
    else:
        # make horizontal mosaic - image and stacks of 3
        n = 3 * (hm.shape[2]//3) + 3
        while hm.shape[2] < n:
            hm = np.dstack((hm,np.zeros(image.shape[:2])))
        vis = image 
        hm = np.uint8(255. * hm)
        for i in range(0,n,3):
            vis = np.hstack((vis, np.dstack((hm[:,:,i],hm[:,:,i+1],hm[:,:,i+2] ) )))

    return vis 

def randomly_drop_visualiztions(project_id, dst_dir = '/tmp/keypoint_heatmap_vis', num = -1, horistack=True):
    # take random frames from the db and show their labeling as gaussian heatmaps
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
    frame_idxs = list(set(frame_idxs))
    shuffle(frame_idxs)

    if num > 0:
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
            vis = vis_heatmap(cv.imread(filepath), keypoint_names, keypoints, horistack = horistack)
            vis_path = os.path.join(dst_dir,'%s.png' % frame_idxs[i] )
            cv.imwrite(vis_path, vis)

if __name__ == '__main__':
    project_id = 1
    randomly_drop_visualiztions(project_id, horistack = False)