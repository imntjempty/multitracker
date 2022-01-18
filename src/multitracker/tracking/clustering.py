
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

def get_temporal_overlap(track1, track2):
    t1s = track1['history_steps'][0]
    t1e = track1['history_steps'][-1]
    t2s = track2['history_steps'][0]
    t2e = track2['history_steps'][-1]
    T1, T2 = [], []
    #print('ABC', len(track1['history']),len(track1['history_steps']),':::',len(track2['history']),len(track2['history_steps']))
    if t2s<=t1s and t2e>=t1e: # track1 completely inside track2
        T1 = track1['history']
        for i in range(len(track2['history'])):
            if track2['history_steps'][i]>=t1s and track2['history_steps'][i]<=t1e:
                T2.append(track2['history'][i])
        
    elif t1s<=t2s and t1e>=t2e: # track2 completely inside track1
        T2 = track2['history']
        for i in range(len(track1['history'])):
            if track1['history_steps'][i]>=t2s and track1['history_steps'][i]<=t2e:
                T1.append(track1['history'][i])
        
    elif t1s<=t2s and t1e>=t2s: # track1 older and leaps into track2
        for i in range(len(track1['history'])):
            if track1['history_steps'][i]>=t2s:
                T1.append(track1['history'][i])
        for i in range(len(track2['history'])):
            if track2['history_steps'][i]<=t1e:
                T2.append(track2['history'][i])
        
    elif t1s>=t2s and t1s<=t2e: # track2 older and leaps into track1
        for i in range(len(track1['history'])):
            if track1['history_steps'][i]<=t2e:
                T1.append(track1['history'][i])
        for i in range(len(track2['history'])):
            if track2['history_steps'][i]>=t1s:
                T2.append(track2['history'][i])
    if len(T1)>0 and len(T2)>0:
        return T1, T2

    return None 

def get_sorted_tracks_by_distance_traveled(tracks):
    distances_traveled = []
    for track in tracks:
        track['traveled'] = 0. 
        for i in range(len(track['history'])-1):
            track['traveled'] = np.linalg.norm( np.array(track['history'][i])-np.array(track['history'][i+1]) )
    traveled_tracks = sorted(tracks, key = lambda k: k['traveled'])[::-1]
    return traveled_tracks

def get_clustlets(tracks):
    clustlets = []

    ## init by selecting expressive tracks (-> longest distance traveled)
    traveled_tracks = get_sorted_tracks_by_distance_traveled(tracks)

    V = [track for track in traveled_tracks if len(track['history'])>4]
    print('[*] starting clustering of tracks with %i/%i tracks' % (len(V),len(tracks)))

    count_iterations = -1
    while len(V) > 0:
        count_iterations+=1
        representative_track = V[0] 
        # calculate similarity to other tracks
        similiar_tracks = []
        for i in range(1, len(V), 1):
        
            # check if temporal overlap
            overlap = get_temporal_overlap(representative_track, V[i])
            if overlap is not None:
                subsegment1, subsegment2 = overlap 
                
                # two tracks are similiar iff direction was consistent troughout the subsegments
                # roughly sample twice every second
                dx1 = subsegment1[0][0] - subsegment1[-1][0]
                dy1 = subsegment1[0][1] - subsegment1[-1][1]
                dx2 = subsegment2[0][0] - subsegment2[-1][0]
                dy2 = subsegment2[0][1] - subsegment2[-1][1]
                L1 = np.sqrt( dx1*dx1 + dy1*dy1 )
                L2 = np.sqrt( dx2*dx2 + dy2*dy2 )
                #angle1 = np.arctan2(dy1,dx1) +2.*np.pi
                #angle2 = np.arctan2(dy2,dx2) +2.*np.pi
                alpha = .5
                sim_angle = .5 * ( dx1*dx2+dy1*dy2 ) / ( 1e-6 +np.sqrt(dx1*dx1+dy1*dy1) * np.sqrt(dx2*dx2+dy2*dy2) )
                sim_dist = 1 - np.abs(L1-L2)/(1e-6+np.max([L1,L2]))
                sim = alpha * sim_angle + (1.-alpha) * sim_dist
                if not np.isnan(sim):
                    delta = 0.6
                    if sim > delta:
                        similiar_tracks.append(V[i])
                else:
                    #print('segment1',subsegment1)
                    #print('segment2',subsegment2)
                    print('rep',len(representative_track['history']),':::',representative_track['history'][0],'->',representative_track['history'][-1])
                    print('tra',len(V[i]['history_steps']),len(V[i]['history']),':::',V[i]['history'][0], V[i]['history'][-1])
                    print(sim, sim_angle,sim_dist,':::',len(subsegment1),len(subsegment2))
                    print()
                
                # two tracks are similiar if maximum minimum distance from track A to B is small
        print(count_iterations,'V',len(V),'sim',len(similiar_tracks))
        clustlets.append( [representative_track] + similiar_tracks )

        # remove representative_track and tracks from V that lie completely inside clustered tracks
        V = V[1:]
        W = []
        min_step, max_step = 1000000,0
        for i in range(len(clustlets[-1])):
            min_step = min(min_step,clustlets[-1][i]['start'])
            max_step = max(min_step,clustlets[-1][i]['end'])
        deletes = []
        for i in range(len(V)):
            if V[i]['start']>=min_step and V[i]['end']<=max_step: # track lies completely in new clustlet
                ''#deletes.append(i)
                pass
            else:
                W.append(V[i])
        V = W 

    return clustlets

def merge_clustlets(clustlets):
    clusters = []
    gamma = 0.2
    # sort ascending by number of trajectories
    clustlets = sorted(clustlets, key = len)[::-1]
    for i, c in enumerate(clustlets[:20]):
        print(i, len(c))
    return clusters

def vis_clusters(clusters):
    import matplotlib.pyplot as plt
    clusters = sorted(clusters, key = len)[::-1]
    for cluster in clusters:
        x = []
        y = []
        for i,track in enumerate(cluster):
            path = np.array(track['history'])
            print('path',path.shape,path.min(),path.max())
            x = path[:,0]
            y = path[:,1]
            colors = ['red','blue','yellow','green','black','gray','cyan','brown','purple']
            c = colors[i%len(colors)]
            plt.plot(x,y,color=c)
        #    x.extend(list(path[:,0]))
        #    y.extend(list(path[:,1]))
        #x, y = np.array(x), np.array(y)
        #plt.plot(x,y)
        #plt.plot(x,yhat, color='red')
        plt.show()

def get_clusters(tracks):
    """
        implements https://publik.tuwien.ac.at/files/PubDat_181914.pdf
    """

    # The first stage of the algorithm is an iterative clustering scheme
    # that groups temporally overlapping trajectories with similar velocity direction and magnitude.
    clustlets = get_clustlets(tracks)
    if 0:
        vis_clusters(clustlets)
    print('[*] found %i clustlets' % len(clustlets))

    # In the second stage (described in Section 3.2) the clusters from
    # the first stage are merged into temporally adjacent clusters covering larger time spans.
    clusters = merge_clustlets(clustlets)    


    return clusters