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
        assign idX
        
"""