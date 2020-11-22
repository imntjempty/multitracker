"""
    group different classed keypoints into individuals

    greedy idea
    =================
    assump: - every individual has one kind of class max 
       min euclidean distance between all pairs of parts belonging to one individual
       make graph
       V: each keypoint 
       E: connects keypoints 
    calc individuals by connected components

    while not all verts assigned to ID:
        find minimum weight edge 
            with shortest distance D between two different classed keypoints A(xa,ya,classA) and B(xb,yb,classB), classA!=classB 
            and where A and/or B is unassigned
            and where D > thresh 
        delete all edges connected to A, that connect to classB nodes and all edges connected to B, that connect to classA nodes 
        assign idX: 
            if A or B has individ, assign both with them
            if no one has individ, assign new id to both of them
        
"""

import numpy as np 
import networkx as nx 

def group_keypoints_into_individuals(keypoints):
    """
        keypoints in [N,X,Y,C(lassID)]
        returns list of integer indiv ids with length N 
    """

    keypoints = np.array(keypoints)
    minp = np.min(keypoints,axis=0)[:2]
    maxp = np.max(keypoints,axis=0)[:2]
    maxD = np.linalg.norm(maxp-minp)

    # append column for indiviv id, init with -1 because all unassigned
    keypoints = np.concatenate((keypoints,-np.ones((keypoints.shape[0],1))),axis=1)

    thresh = 0.2
    count_iterations = 0
    count_indiv = 0
    done = False
    # while at least one keypoint has no id assignment
    while not done:
        # find minimum distance between any two keypoints with different classes, where at least one of them has no class assignment
        min_dist, min_idx = 1e6, None 
        for i,[xa,ya,ca,idva] in enumerate( keypoints ):
            if i > 0:
                for j in range(i):
                    xb,yb,cb,idvb = keypoints[j]
                    if not ca==cb: # different class
                        if idva < 0 or idvb < 0: # not both already assigned
                            D = np.sqrt((xa-xb)**2+(ya-yb)**2)
                            if D / maxD < thresh: # if two keypoints not to far apart 
                                if D < min_dist:
                                    min_dist = D 
                                    min_idx = (i,j)
        # id assignment
        if min_idx is not None:
            if keypoints[min_idx[0]][3] >= 0: # if A already set, set B 
                keypoints[min_idx[1]][3] = keypoints[min_idx[0]][3]
            elif keypoints[min_idx[1]][3] >= 0: # if B already set, set A 
                keypoints[min_idx[0]][3] = keypoints[min_idx[1]][3]
            elif keypoints[min_idx[0]][3]<0 and keypoints[min_idx[1]][3]<0:
                # set up new indivi
                keypoints[min_idx[0]][3] = count_indiv
                keypoints[min_idx[1]][3] = count_indiv
                count_indiv += 1 
        else:
            # global minimum found
            done = True 

        # continue if not all keypoints assigned
        done = done or np.min(keypoints[:,3])>=0

    # return list of indiv ids 
    return keypoints



