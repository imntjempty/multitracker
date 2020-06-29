"""
    graph track

    http://openaccess.thecvf.com/content_cvpr_2018/papers/Girdhar_Detect-and-Track_Efficient_Pose_CVPR_2018_paper.pdf

    We represent these detections in a graph, where each detected bounding box (representing a person) in a
    frame becomes a node. We define edges to connect each box in a frame to every box in the next frame. The cost of
    each edge is defined as the negative likelihood of the two boxes linked on that edge to belong to the same person.

    We initialize tracks on the first frame and any boxes that do not get matched to an existing track instantiate a new track.

    We start from the highest confidence match, select that edge and remove the two connected nodes out of consideration. This 
    process of connecting each predicted box in the current frame with previous frame is repeatedly applied from the first to 
    the last frame of the video.

"""


import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py

from multitracker import util 
from multitracker.keypoint_detection import heatmap_drawing, model 
from multitracker.keypoint_detection import predict
from multitracker.graph_tracking import group_indiv

def load_model(path_model):
    t0 = time.time()
    trained_model = tf.keras.models.load_model(h5py.File(path_model, 'r'))
    t1 = time.time()
    print('[*] loaded model from %s in %f seconds.' %(path_model,t1-t0))
    return trained_model 

def load_data(project_id,video_id):
    frames_dir = predict.get_project_frame_train_dir(project_id, video_id)
    frame_files = sorted(glob(os.path.join(frames_dir,'*.png')))
    
    #frame_files = frame_files[int(np.random.uniform(2000)):]
    max_minutes = [0,2][0]
    if max_minutes >0:
        frame_files = frame_files[: 60*max_minutes*30 ]

    if len(frame_files) == 0:
        raise Exception("ERROR: no frames found in " + str(frames_dir))
    print('[*] found %i frames' % len(frame_files))
    return frame_files

def get_heatmaps_keypoints(heatmaps):
    keypoints = [] 
    for c in range(heatmaps.shape[2]-1): # dont extract from background channel
        channel_candidates = predict.extract_frame_candidates(heatmaps[:,:,c], thresh = 0.5, pp = int(0.02 * np.min(heatmaps.shape[:2])))
        for [px,py,val] in channel_candidates:
            keypoints.append([px,py,c])

    # [debug] filter for nose 
    #keypoints = [kp for kp in keypoints if kp[2]==1]
    return keypoints 

def inference_heatmap(config, trained_model, frame):
    # predict whole image, height like trained height and variable width 
    # to keep aspect ratio and relative size        
    bs, config['n_inferences'] = 1, 1
    #bs, config['n_inferences'] = 2, 3
    w = 1+int(2*config['img_height']/(float(frame.shape[0]) / frame.shape[1]))
    xsmall = cv.resize(frame, (w,2*config['img_height']))
    xsmall = tf.expand_dims(tf.convert_to_tensor(xsmall),axis=0)
    
    if bs > 1: 
        xsmall = tf.tile(xsmall,[bs,1,1,1])

    # 1) inference: run trained_model to get heatmap predictions
    tsb = time.time()
    if config['n_inferences'] == 1:
        y = trained_model.predict(xsmall)[-1]
    else:
        y = trained_model(xsmall, training=True)[-1] / config['n_inferences']
        for ii in range(config['n_inferences']-1):
            y += trained_model(xsmall, training=True)[-1] / config['n_inferences']
    if bs > 1:
        y = tf.reduce_mean(y,axis=[0]) # complete batch is of same image, so second dim average
    else:
        y = y[0,:,:,:]
    
    if config['n_inferences'] > 1:
        y = y.numpy()
    
    y = cv.resize(y,tuple(frame.shape[:2][::-1]))
    tse = time.time() 
    return y

colors = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (255,0,128),
        (255,0,255),
        (0,255,255),
        (128,128,0),
        (255,255,0),
        (128,0,128),
        (0,128,128),
        (128,128,255),
        (255,128,128),
        (128,255,128),
        (255,255,128),
        (255,128,255),
        (128,255,255),
        (64,196,255),
        (255,64,196),
        (64,255,196),
        (196,64,255),
        (255,64,196),
        (196,255,64),
    ]

def get_keypoints_vis(frame, keypoints, keypoint_names):
    vis_keypoints = np.zeros(frame.shape,'uint8')
    
    # draw circles
    for [x,y,class_id,indv ] in keypoints:
        radius = np.min(vis_keypoints.shape[:2]) // 200
        px,py = np.int32(np.around([x,y]))
        # color by indv
        color = colors[int(indv)%len(colors)]
        c1,c2,c3 = color
        vis_keypoints = cv.circle(vis_keypoints,(px,py),radius,(int(c1),int(c2),int(c3)),-1)
    
    # draw labels
    for [x,y,class_id,indv ] in keypoints:
        px,py = np.int32(np.around([x,y]))
        color = colors[int(indv%len(colors))]
        name = "%i %s"%(int(indv),keypoint_names[int(class_id)])
        #cv.putText( vis_keypoints, name, (px+3,py-8), cv.FONT_HERSHEY_COMPLEX, 1, color, 3 )
    
    vis_keypoints = np.uint8(vis_keypoints//2 + frame//2)
    return vis_keypoints 

cnt_total_tracks = 0
cnt_total_individuals = 0
def update_tracks(individuals, tracks, keypoints, max_dist = 100):
    global cnt_total_tracks
    global cnt_total_individuals
    '''
        update tracks
        while not all tracks matched:
            find minimum distance between unmatched track and unmatched keypoint
            add keypoint to history of track
    '''

    #keypoints = [kp for kp in keypoints]
    matched_keypoints = []
    delete_tracks = []
    
    if len(tracks) > 0:
        tracks_matched = np.zeros((len(tracks)))
        for i in range(len(tracks)):
            min_dist, min_idx = 1e6, (-1,-1)
            for j in range(len(tracks)):
                ts = tracks[j]['position']
                # if track not matched 
                if tracks_matched[j] == 0:
                    for k in range(len(keypoints)):
                        # if classes match 
                        if tracks[j]['history_class'][-1] == keypoints[k][2]:
                            # if keypoint not already matched
                            if k not in matched_keypoints:
                                D = np.sqrt( (ts[0]-keypoints[k][0])**2 + (ts[1]-keypoints[k][1])**2 )
                                if D < min_dist and D < max_dist:
                                    min_dist = D   
                                    min_idx = (j,k)
            # match min distance keypoint to track 
            if min_idx[0] >= 0:
                alpha = 0.9
                tracks[min_idx[0]]['history'].append(keypoints[min_idx[1]][:2])#tracks[min_idx[0]]['position'])
                tracks[min_idx[0]]['history_class'].append(keypoints[min_idx[1]][2])
                estimated_pos = np.array(tracks[min_idx[0]]['position'])#
                #if len(tracks[min_idx[0]]['history'])>2:
                #    estimated_pos = estimated_pos + ( np.array(tracks[min_idx[0]]['history'][-2]) -np.array(keypoints[min_idx[1]][:2])  )
                if i>0:
                    print(i, tracks[min_idx[0]]['position'], '<->',keypoints[min_idx[1]][:2],':::',tracks[min_idx[0]]['history_class'][-2],keypoints[min_idx[1]][2], 'D',np.sqrt( (tracks[min_idx[0]]['position'][0]-keypoints[min_idx[1]][0])**2 + (tracks[min_idx[0]]['position'][1]-keypoints[min_idx[1]][1])**2 ))
                
                tracks[min_idx[0]]['position'] =  alpha * estimated_pos + (1-alpha) * np.array(keypoints[min_idx[1]][:2])
                tracks[min_idx[0]]['position'] = np.array(keypoints[min_idx[1]][:2])
                tracks_matched[min_idx[0]] = 1 
                matched_keypoints.append(min_idx[1])

                        
        
        # for each unmatched track: increase num_misses 
        for i,track in enumerate(tracks):
            if tracks_matched[i] == 0:
                tracks[i]['num_misses'] += 1 
                if tracks[i]['num_misses'] > 100: # lost track over so many consecutive frames
                    # delete later
                    if i not in delete_tracks:
                        delete_tracks.append(i)
                if tracks[i]['age'] < 4: # miss already at young age are false positives
                    if i not in delete_tracks:
                        delete_tracks.append(i)
            else:
                tracks[i]['num_misses'] = 0

    # for each unmatched keypoint: create new track
    max_intra_indiv_dist = 250

    for k in range(len(keypoints)):
        if not k in matched_keypoints:
            # create new track and find idv to assign
            idv = - 1
            min_dist, min_idx = 1e6, -1 
            for j, track in enumerate(tracks):
                #
                D = np.sqrt( (track['position'][0]-keypoints[k][0])**2 + (track['position'][1]-keypoints[k][1])**2 )
                if D < min_dist: # new min distance found
                    if D < max_intra_indiv_dist: # not too far away
                        # only add if min dist indiv not already has keypoint with same class as k 
                        has_already_same_class_item = False 
                        for other_track in tracks:
                            if other_track['idv'] == track['idv']:
                                has_already_same_class_item = has_already_same_class_item or other_track['history_class'][-1] == keypoints[k][2]
                        if not has_already_same_class_item:
                            min_dist = D 
                            min_idx = j 
            if min_idx >= 0:
                # found existing track to assign itself to
                idv = tracks[min_idx]['idv']
            else:
                # no sutiable indiv found, have to setup a new one
                cnt_total_individuals += 1
                idv = cnt_total_individuals

            new_track = {
                'id': cnt_total_tracks,
                'history': [keypoints[k][:2]],
                'position': keypoints[k][:2],
                'history_class': [keypoints[k][2]],
                'num_misses': 0,
                'age': 0,
                'idv': idv
            }
            tracks.append(new_track)
            cnt_total_tracks += 1

    # update age of all tracks 
    for track in tracks:
        track['age'] += 1

    # delete tracks
    for i in delete_tracks[::-1]:
        del tracks[i]
    return individuals, tracks 

def draw_tracks(frame, individuals, tracks):
    """
        paint history path for each indiv mass center
        paint 'sceleton' for each indiv keypoints -> each idv unique color
        paint keypoints in class keypoint colors
        only draw keypoints assigned to idv
    """
    vis = np.array(frame,copy=True)
    def draw_legend():
        # draw legend on right side showing colors and names of keypoint types
        pass
    draw_legend()

    for k,track in enumerate(tracks):
        radius = 30
        #c1,c2,c3 = colors[track['id']%len(colors)]
        
        # calc history of past center points 

        # draw history as line for each history point connected to prev point
        for i in range(len(track['history'])):
            if i > 0:
                c1,c2,c3 = colors[track['history_class'][i]%len(colors)]
                p1 = tuple(np.int32(np.around(track['history'][i])))
                p2 = tuple(np.int32(np.around(track['history'][i-1])))
                vis = cv.line(vis,p1,p2,(int(c1),int(c2),int(c3)),2)
                #vis = cv.circle(vis,(px,py),radius,(int(c1),int(c2),int(c3)),-1)
        if len(track['history'])>0:
            # draw last segment
            c1,c2,c3 = colors[track['history_class'][-1]%len(colors)]
            plast = tuple(np.int32(np.around(track['history'][-1])))
            ppos = tuple(np.int32(np.around(track['position'])))
            vis = cv.line(vis,plast,ppos,(int(c1),int(c2),int(c3)),2)
                
            # draw actual position
            # draw text label 
            #name = chr(65+track['idv'])
            name = str(track['idv'])
            cv.putText( vis, name, (ppos[0]+3,ppos[1]-8), cv.FONT_HERSHEY_COMPLEX, 0.75, (int(c1),int(c2),int(c3)), 2 )

    
    return vis 

def track(config, model_path, project_id, video_id):
    project_id = int(project_id)
    video_id = int(video_id)
    output_dir = os.path.expanduser('~/data/multitracker/tracks/%i/%i/%s' % (project_id, video_id, model_path.split('/')[-1].split('.')[0]))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # load pretrained model
    trained_model = load_model(model_path)

    # load frame files
    frame_files = load_data(project_id,video_id)
    config['input_image_shape'] = cv.imread(frame_files[0]).shape[:2]

    tracks = [] 
    individuals = []

    for i, frame_file in enumerate(frame_files):
        tframestart = time.time()
        fnameo = os.path.join(output_dir,frame_file.split('/')[-1])
        # inference frame
        frame = cv.imread(frame_file)
        heatmaps = inference_heatmap(config, trained_model, frame)
        # detect keypoints
        keypoints = get_heatmaps_keypoints(heatmaps)
        # update tracks
        individuals, tracks = update_tracks(individuals, tracks, keypoints)
        vis_tracks = draw_tracks(frame, individuals, tracks)
        cv.imwrite(fnameo,vis_tracks)
        #cv.imshow('track result',vis_tracks)
        #cv.waitKey(30)

        if 0:#i==0: #if len(tracks) == 0 and len(keypoints) >0:
            # init keypoints into individuals
            #keypoints = group_indiv.group_keypoints_into_individuals(keypoints)
            vis_keypoints = get_keypoints_vis(frame, keypoints, config['keypoint_names'])
            cv.imwrite(fnameo,vis_keypoints)
            print('[*] wrote',fnameo)
        

        tframeend = time.time()
        eta_min = (len(frame_files)-i) * (tframeend - tframestart) / 60.
        print('[* %i/%i]'%(i,len(frame_files)), fnameo ,'heatmaps',heatmaps.shape,'min/max %f/%f'%(heatmaps.min(),heatmaps.max()),'found %i keypoints'%len(keypoints),'in %f seconds. estimated %f minutes remaining'%(tframeend - tframestart,eta_min))
    
    if 1:
        util.make_video(output_dir, output_dir+'.mp4',"%05d.png")
    
    return tracks 
 

def main(args):
    config = model.get_config(project_id = args.project_id)

    tracks = track(config, args.model, args.project_id, args.video_id)

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    main(args)