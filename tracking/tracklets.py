
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
from multitracker.tracking.inference import load_data

max_dist_keypoint = 75 # max distance a keypoint can travel from one frame to the other
max_num_misses = 100
max_age_wo_detection = 4
max_steps_wo_movement = 7

def update(alive_tracks, dead_tracks, keypoints, step, max_dist = max_dist_keypoint):
    #keypoints = [kp for kp in keypoints]
    matched_keypoints = []
    delete_tracks = []
    tracks = alive_tracks
    if len(tracks) > 0:
        tracks_matched = np.zeros((len(tracks)))
        for i in range(len(tracks)):
            if tracks[i]['alive']:
                min_dist, min_idx = 1e6, (-1,-1)
                for j in range(len(tracks)):
                    ts = tracks[j]['position']
                    # if track not matched 
                    if tracks[j]['alive'] and tracks_matched[j] == 0:
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
                    tracks[min_idx[0]]['history_steps'].append(step)
                    estimated_pos = np.array(tracks[min_idx[0]]['position'])#
                    #if len(tracks[min_idx[0]]['history'])>2:
                    #    estimated_pos = estimated_pos + ( np.array(tracks[min_idx[0]]['history'][-2]) -np.array(keypoints[min_idx[1]][:2])  )
                    if i>0 and 0:
                        print(i, tracks[min_idx[0]]['position'], '<->',keypoints[min_idx[1]][:2],':::',tracks[min_idx[0]]['history_class'][-2],keypoints[min_idx[1]][2], 'D',np.sqrt( (tracks[min_idx[0]]['position'][0]-keypoints[min_idx[1]][0])**2 + (tracks[min_idx[0]]['position'][1]-keypoints[min_idx[1]][1])**2 ))
                    
                    tracks[min_idx[0]]['position'] =  alpha * estimated_pos + (1-alpha) * np.array(keypoints[min_idx[1]][:2])
                    tracks[min_idx[0]]['position'] = np.array(keypoints[min_idx[1]][:2])
                    tracks_matched[min_idx[0]] = 1 
                    matched_keypoints.append(min_idx[1])  
        
        # for each unmatched track: increase num_misses 
        for i,track in enumerate(tracks):
            if tracks_matched[i] == 0:
                tracks[i]['num_misses'] += 1 
                if tracks[i]['num_misses'] > max_num_misses: # lost track over so many consecutive frames
                    # delete later
                    if i not in delete_tracks:
                        delete_tracks.append(i)
                if tracks[i]['age'] < max_age_wo_detection: # miss already at young age are false positives
                    if i not in delete_tracks:
                        delete_tracks.append(i)
            else:
                tracks[i]['num_misses'] = 0

    for k in range(len(keypoints)):
        if not k in matched_keypoints:
            
            new_track = {
                'id': len(tracks),
                'history': [keypoints[k][:2]],
                'position': keypoints[k][:2],
                'history_class': [keypoints[k][2]],
                'num_misses': 0,
                'num_wo_movement': 0,
                'age': 0,
                'start': step,
                'end': None,
                'alive': True,
                'history_steps': [step]
            }
            tracks.append(new_track)
            
     # update age of all tracks 
    for track in tracks:
        track['age'] += 1

    # delete tracks
    delete_tracks = sorted(delete_tracks)
    for i in delete_tracks[::-1]:
        alive_tracks[i]['end'] = step
        alive_tracks[i]['alive'] = False
        # only save tracks that lasted longer than half a second
        #print(step,i,alive_tracks[i]['end'],alive_tracks[i]['start'])
        if alive_tracks[i]['end'] is not None and alive_tracks[i]['end']-alive_tracks[i]['start']  > int(0.5 * 30):
            dead_tracks.append(alive_tracks[i])
        del alive_tracks[i]
    return alive_tracks, dead_tracks


def draw_tracks(frame, tracks, config):
    """
        paint history path for each indiv mass center
        paint 'sceleton' for each indiv keypoints -> each idv unique color
        paint keypoints in class keypoint colors
        only draw keypoints assigned to idv
    """
    colors = util.get_colors()
    vis = np.array(frame,copy=True)
    def draw_legend():
        # draw legend on right side showing colors and names of keypoint types
        lw = frame.shape[1] // 6
        lh = len(config['keypoint_names'])*lw//4
        legend = 128 * np.ones((lh,lw,3),'uint8')
        pad = lw // 10
        loffx = 6 * lw // 10
        for i,kpname in enumerate(config['keypoint_names']):
            ly = i * lh // len(config['keypoint_names'])
            legend = cv.putText(legend,config['keypoint_names'][i].replace('_',' '),(pad,pad+ly+35),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0))
            legend = cv.rectangle(legend,(pad+loffx,pad+ly),(lw-pad,(i+1) * lh // len(config['keypoint_names'])),[int(cc) for cc in colors[i]],-1)
        legend = cv.cvtColor(legend, cv.COLOR_BGR2RGB)
        # paste legend onto frame
        vis[0:legend.shape[0],-legend.shape[1]:,:] = legend 

    color_indv = (128,255,128)
    
        
    if 1:
        for k,track in enumerate(tracks):
            radius = 30
            #c1,c2,c3 = colors[track['id']%len(colors)]
            
            # calc history of past center points 

            if track['alive']:
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
    
    if 0:
        for idv in individuals.keys():
            ## draw connected keypoints
            p1 = tuple(np.int32(np.around(individuals[idv]['history'][-1])))    
            for track in tracks:
                if track['idv'] == idv:
                    c1,c2,c3 = colors[track['history_class'][-1]%len(colors)]
                    p2 = tuple(np.int32(np.around(track['history'][-1])))    
                    vis = cv.line(vis,p1,p2,(int(c1),int(c2),int(c3)),2)

    if 0:
        for idv in individuals.keys():
            ## draw text label for each indiviudial
            color_indv = [int(cc) for cc in colors[(idv+len(config['keypoint_names']))%len(colors)]]
            name = str(idv)
            # draw at beginning of history
            ppos = tuple(np.int32(np.around(individuals[idv]['history'][0])))
            vis = cv.putText( vis, name, (ppos[0]+3,ppos[1]-8), cv.FONT_HERSHEY_COMPLEX, 0.75, color_indv, 2 )
            # draw at end of history
            if len(individuals[idv]['history'])>1:
                ppos = tuple(np.int32(np.around(individuals[idv]['history'][-1])))
                vis = cv.putText( vis, name, (ppos[0]+3,ppos[1]-8), cv.FONT_HERSHEY_COMPLEX, 0.75, color_indv, 2 )

    draw_legend()
    
    return vis 


def get_tracklets(detections, model_path, config, project_id, video_id, should_vis = False):
    '''
        update tracks
        while not all tracks matched:
            find minimum distance between unmatched track and unmatched keypoint
            add keypoint to history of track
    '''
    ttrackstart = time.time()
    project_id = int(project_id)
    video_id = int(video_id)
    output_dir = os.path.expanduser('~/data/multitracker/tracks/%i/%i/%s' % (project_id, video_id, model_path.split('/')[-1].split('.')[0]))
    output_file = output_dir + '.avi'
    if should_vis and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if should_vis and os.path.isfile(output_file):
        os.remove(output_file)

    # load frame files
    frame_files = load_data(project_id,video_id)
    config['input_image_shape'] = cv.imread(frame_files[0]).shape[:2]

    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    writer = cv.VideoWriter(output_file, fourcc, 30, (config['input_image_shape'][1], config['input_image_shape'][0]), True)
    
    alive_tracks, dead_tracks = [], []

    frame_files = frame_files[:10000]

    for i, frame_file in enumerate(frame_files):
        tstart = time.time()
        alive_tracks, dead_tracks = update(alive_tracks, dead_tracks, detections[frame_file], i)
        if should_vis:    
            vis = draw_tracks(cv.imread(frame_file), alive_tracks, config)
            if writer is not None:
                writer.write(vis)
        minutes_left = 1+int((len(frame_files)-i)*(time.time() - tstart)/60.)
        print('[*] tracklets: %i/%i frames, %i/%i tracklets. ETA %i minutes remaining' % (i,len(frame_files),len(alive_tracks),len(dead_tracks),minutes_left))
        
    # after video kill all alive tracks 
    for t in range(len(alive_tracks)):
        alive_tracks[t]['alive'] = False 
        alive_tracks[t]['end'] = len(frame_files)

    if writer is not None:
	    writer.release()

    ttrackend = time.time()
    print('[*] finished tracking after %i minutes. saved to %s' % (int((ttrackend-ttrackstart)/60.),output_file))

    return alive_tracks + dead_tracks