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
from multitracker.tracking import inference
from multitracker.graph_tracking import group_indiv
from multitracker.be import dbconnection

max_intra_indiv_dist = 250 # max distance a track and a keypoint can have to be considered belonging to the same individuum
max_dist_keypoint = 75 # max distance a keypoint can travel from one frame to the other
max_num_misses = 100
max_age_wo_detection = 4
max_steps_wo_movement = 7
min_dist_movement = 2
thresh_detection = 0.5
max_minutes = 1#3

def load_model(path_model):
    t0 = time.time()
    trained_model = tf.keras.models.load_model(h5py.File(path_model, 'r'))
    t1 = time.time()
    print('[*] loaded model from %s in %f seconds.' %(path_model,t1-t0))
    return trained_model 

def load_data(project_id,video_id):
    frames_dir = inference.get_project_frame_train_dir(project_id, video_id)
    frame_files = sorted(glob(os.path.join(frames_dir,'*.png')))
    
    #frame_files = frame_files[int(np.random.uniform(2000)):]
    if max_minutes >0:
        nn = int(60*max_minutes*30 )
        ns = int(np.random.random()*(len(frame_files)-nn))
        #frame_files = frame_files[ ns:ns+nn ]
        frame_files = frame_files[:nn]

    if len(frame_files) == 0:
        raise Exception("ERROR: no frames found in " + str(frames_dir))
    print('[*] found %i frames' % len(frame_files))
    return frame_files

def get_heatmaps_keypoints(heatmaps):
    keypoints = [] 
    for c in range(heatmaps.shape[2]-1): # dont extract from background channel
        channel_candidates = inference.extract_frame_candidates(heatmaps[:,:,c], thresh = thresh_detection, pp = int(0.02 * np.min(heatmaps.shape[:2])))
        for [px,py,val] in channel_candidates:
            keypoints.append([px,py,c])

    # [debug] filter for nose 
    #keypoin1s = [kp for kp in keypoints if kp[2]==1]
    return keypoints 

def inference_heatmap(config, trained_model, frame):
    # predict whole image, height like trained height and variable width 
    # to keep aspect ratio and relative size        
    #bs, config['n_inferences'] = 1, 1
    bs, config['n_inferences'] = 4, 1
    w = 1+int((1./config['fov'])*config['img_height']/(float(frame.shape[0]) / frame.shape[1]))
    xsmall = cv.resize(frame, (w,int((1./config['fov'])*config['img_height'])))
    xsmall = tf.expand_dims(tf.convert_to_tensor(xsmall),axis=0)
    
    if bs > 1: 
        xsmall = tf.tile(xsmall,[bs,1,1,1])

    # 1) inference: run trained_model to get heatmap predictions
    tsb = time.time()
    if config['n_inferences'] == 1 and bs == 1:
        y = trained_model.predict(xsmall)[-1]
    else:
        y = trained_model(xsmall, training=True)[-1] / config['n_inferences']
        for ii in range(config['n_inferences']-1):
            y += trained_model(xsmall, training=True)[-1] / config['n_inferences']
    y = tf.reduce_mean(y,axis=[0]) # complete batch is of same image, so second dim average
    
    #if config['n_inferences'] > 1:
    try:
        y = cv.resize(y,tuple(frame.shape[:2][::-1]))
    except:
        y = y.numpy()
        y = cv.resize(y,tuple(frame.shape[:2][::-1]))
    #print('y',y.shape,frame.shape)
    
    tse = time.time() 
    return y

colors = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (255,0,255),
        (255,255,0),
        (128,128,0),
        (255,255,0),
        (128,0,128),
        (0,128,128),
        (128,128,255),
        (255,128,128),
        (255,0,128),
        (255,255,128),
        (255,128,255),
        (128,255,255),
        (64,196,255),
        (255,64,196),
        (64,255,196),
        (196,64,255),
        (128,255,128),
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
def update_tracks(individuals, tracks, keypoints, max_dist = max_dist_keypoint):
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

    # for each unmatched keypoint: create new track
    

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
                        #if not has_already_same_class_item:
                        min_dist = D 
                        min_idx = j 
            if min_idx >= 0:
                # found existing track to assign itself to
                idv = tracks[min_idx]['idv']
            else:
                # no suitable indiv found, have to setup a new one
                idv = cnt_total_individuals
                individuals[idv] = {'history':[],'num_misses':0}
                cnt_total_individuals += 1
                
            new_track = {
                'id': cnt_total_tracks,
                'history': [keypoints[k][:2]],
                'position': keypoints[k][:2],
                'history_class': [keypoints[k][2]],
                'num_misses': 0,
                'num_wo_movement': 0,
                'age': 0,
                'idv': idv
            }
            tracks.append(new_track)
            cnt_total_tracks += 1

    # check if track has moved
    for i in range(len(tracks)):
        if len(tracks[i]['history'])>1 and np.linalg.norm( np.array(tracks[i]['history'][-2])- np.array(tracks[i]['history'][-1] )) < min_dist_movement:
            tracks[i]['num_wo_movement'] += 1
        else:
            tracks[i]['num_wo_movement'] = 0

        if tracks[i]['num_wo_movement'] > max_steps_wo_movement:
            if i not in delete_tracks:
                delete_tracks.append(i)

    # update age of all tracks 
    for track in tracks:
        track['age'] += 1

    # delete tracks
    delete_tracks = sorted(delete_tracks)
    for i in delete_tracks[::-1]:
        del tracks[i]

    '''## check for each matched keypoint if it should belong to another individuum. if the other already has such a keypoint, calc swap cost. if it's free just take it
    for i, track in enumerate(tracks):
        for j, other_track in enumerate(tracks):
            if i<j: # symmetrical operation, upper triangle
                if track['history_class'][-1] == other_track['history_class'][-1]: # only switch same classed
                    # check if swap has 
        # calc shortest distance between track and other track idv's keypoints and vis versa 
                    idx1, dist1 = get_nearest_neighbor(track, other_tracks)
                    idx2, dist2 = get_nearest_neighbor(other_track, tracks)
                    idx3, dist3 = get_nearest_neighbor(track, tracks)
                    idx4, dist4 = get_nearest_neighbor(other_track, other_tracks)
                    #if not track['idv'] == other_track['idv'] and track['history_class'][-1] == other_track['history_class'][-1]: # different ids but same class '''
                    
                
    # split up individuals if distance between two keypoints are too high
    def get_nearest_neighbor(ob, obs, different_classes=False, same_idv=False,different_idv=False):
        min_dist, min_idx = 1e6, -1
        for j in range(len(obs)):
            ts = obs[j]['history'][-1]
            # if track not matched 
            
            D = np.sqrt( (ts[0]-ob['history'][-1][0])**2 + (ts[1]-ob['history'][-1][1])**2 )
            if D > 0 and D < min_dist:
                if not different_classes or (different_classes and not ob['history_class'][-1] == obs[j]['history_class'][-1]): # different classes!
                    if not same_idv or (same_idv and ob['idv'] == obs[j]['idv']): # different classes!
                        if not different_idv or (different_idv and not ob['idv'] == obs[j]['idv']): # different classes!
                            min_dist = D   
                            min_idx = j 
        return min_idx, min_dist 

    # if track is too far away from other indiviuals keypoints , check for another one close by and set up a new one
    for i in range(len(tracks)):
        track = tracks[i]
        min_dist, min_idx = get_nearest_neighbor(track, tracks, different_classes=True, same_idv = True)
        if min_dist > max_intra_indiv_dist:
            # has has close neighbor with other tracks?
            min_dist_oidv, min_idx_oidv = get_nearest_neighbor(track, tracks, different_classes=True, different_idv=True)
            if min_dist_oidv < min_dist:
                print('[*] switching over track',i,'idv from %s to %s' % (tracks[i]['idv'],tracks[min_idx_oidv]['idv']))
                tracks[i]['idv'] = tracks[min_idx_oidv]['idv']
                
            else:
                print('[*] lost indiv connection, new track idv',cnt_total_individuals)
                tracks[i]['idv'] = cnt_total_individuals
                individuals[cnt_total_individuals] = {'history':[],'num_misses':0}
                cnt_total_individuals += 1

    # calc new position for indiv based on detected and assigned keypoints of current frame 
    individual_keypoints = {}
    for k in range(len(tracks)):
        track = tracks[k]
        if not track['idv'] in individual_keypoints:
            individual_keypoints[track['idv']] = {'history':[]}
        individual_keypoints[track['idv']]['history'].append(track['history'][-1])
    
    for k in individual_keypoints.keys():
        if len(individual_keypoints[k]['history']) > 0:
            individual_keypoints[k]['history'] = np.array(individual_keypoints[k]['history'])
            c = np.mean(individual_keypoints[k]['history'],axis=0)
            bbb = 0.99
            #if len(individuals[k]['history'])>0:
            #    c = bbb*individuals[k]['history'][-1] + (1-bbb)*c 
            individuals[k]['history'].append(c)
            individuals[k]['num_misses'] = -1 
        
        
    # delete indiv if for so long not seen
    keys = list(individuals.keys())
    for k in keys:
        individuals[k]['num_misses'] += 1
        if individuals[k]['num_misses'] > 100:
            del individuals[k]
    return individuals, tracks 


def draw_tracks(frame, individuals, tracks, config):
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
    
        
    # draw history of individuals movements
    for idv in individuals.keys():
        #c = colors[(idv + len(config['keypoint_names'])%len(colors))]
        c = (128,128,128)
        color_indv = [int(cc) for cc in colors[(idv+len(config['keypoint_names']))%len(colors)]]
        if len(individuals[idv]['history'])>1:
            for j in range(1,len(individuals[idv]['history']),1):
                p1 = tuple(np.int32(np.around(individuals[idv]['history'][j])))
                p2 = tuple(np.int32(np.around(individuals[idv]['history'][j-1])))
                vis = cv.line(vis,p1,p2,color_indv,3)
        
    if 1:
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
    
    if 0:
        for idv in individuals.keys():
            ## draw connected keypoints
            p1 = tuple(np.int32(np.around(individuals[idv]['history'][-1])))    
            for track in tracks:
                if track['idv'] == idv:
                    c1,c2,c3 = colors[track['history_class'][-1]%len(colors)]
                    p2 = tuple(np.int32(np.around(track['history'][-1])))    
                    vis = cv.line(vis,p1,p2,(int(c1),int(c2),int(c3)),2)

    if 1:
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

def track(config, model_path, project_id, video_id):
    project_id = int(project_id)
    video_id = int(video_id)
    output_dir = os.path.join( dbconnection.base_data_dir, 'tracks/%i/%i/%s' % (project_id, video_id, model_path.split('/')[-1].split('.')[0]))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # load pretrained model
    trained_model = load_model(model_path)

    # load frame files
    frame_files = load_data(project_id,video_id)
    config['input_image_shape'] = cv.imread(frame_files[0]).shape[:2]

    tracks = [] 
    individuals = {}

    for i, frame_file in enumerate(frame_files):
        tframestart = time.time()
        fnameo = os.path.join(output_dir,frame_file.split('/')[-1])
        # inference frame
        #print('frame_file',frame_file)
        frame = cv.imread(frame_file)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        tread = time.time()
        heatmaps = inference_heatmap(config, trained_model, frame)
        tinference = time.time()
        # detect keypoints
        keypoints = get_heatmaps_keypoints(heatmaps)
        tkeypoints = time.time()
        # update tracks
        individuals, tracks = update_tracks(individuals, tracks, keypoints)
        ttracking = time.time()
        vis_tracks = draw_tracks(frame, individuals, tracks, config)
        tvis = time.time()
        cv.imwrite(fnameo,vis_tracks)
        twrite = time.time()
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
            print('[* timings] ioread %f, inference %f, keypoints %f, tracks %f, vis %f, write %f'%(tread-tframestart,tinference-tread,tkeypoints-tinference,ttracking-tkeypoints,tvis-ttracking,twrite-tvis))
    
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