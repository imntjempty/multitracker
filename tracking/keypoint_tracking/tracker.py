import numpy as np 

class KeypointTrack(object):
    def __init__(self, id, keypoint_pos, keypoint_class ):
        self.id = id 
        self.history = [keypoint_pos]
        self.position = keypoint_pos 
        self.history_class = [keypoint_class]
        self.history_estimated = [keypoint_pos]
        self.num_misses = 0 
        self.num_wo_movement = 0 
        self.age = 0 
        self.alive = True 
        #self.idv 
        
    
    def age_1step(self):
        self.age += 1 

class KeypointTracker(object):
    def __init__(self):
        self.tracks = []
        self.graveyard = []
        self.max_num_misses = 100
        self.max_age_wo_detection = 2
        self.max_dist_keypoint = 75

    def get_deadnalive_tracks(self):
        return self.graveyard + self.tracks 

    def match_keypoints_and_update_tracks(self, keypoints):
        matched_keypoints, delete_tracks = [], []
        
        if len(self.tracks) > 0:
            tracks_matched = np.zeros((len(self.tracks)))
            for k in range(len(keypoints)):
                min_dist, min_idx = 1e6, (-1,-1)
                for j in range(len(self.tracks)):
                    ts = self.tracks[j].position
                    # if track not matched 
                    if tracks_matched[j] == 0:
                        # if classes match 
                        if self.tracks[j].history_class[-1] == keypoints[k][2]:
                            # if keypoint not already matched
                            if k not in matched_keypoints:
                                D = np.sqrt( (ts[0]-keypoints[k][0])**2 + (ts[1]-keypoints[k][1])**2 )
                                if D < min_dist and D < self.max_dist_keypoint:
                                    min_dist = D   
                                    min_idx = (j,k)
                # match min distance keypoint to track 
                if min_idx[0] >= 0:
                    alpha = 0.1
                    self.tracks[min_idx[0]].history.append(keypoints[min_idx[1]][:2])#tracks[min_idx[0]]['position'])
                    self.tracks[min_idx[0]].history_class.append(keypoints[min_idx[1]][2])
                    estimated_pos = np.array(self.tracks[min_idx[0]].position)#
                    #if len(tracks[min_idx[0]].history)>2:
                    #    estimated_pos = estimated_pos + ( np.array(tracks[min_idx[0]]['history'][-2]) -np.array(keypoints[min_idx[1]][:2])  )
                    if len(self.tracks[min_idx[0]].history_class)>1 and 1:
                        print(k, j, self.tracks[min_idx[0]].position, '<->',keypoints[min_idx[1]][:2],':::',self.tracks[min_idx[0]].history_class[-2],keypoints[min_idx[1]][2], 'D',np.sqrt( (self.tracks[min_idx[0]].position[0]-keypoints[min_idx[1]][0])**2 + (self.tracks[min_idx[0]].position[1]-keypoints[min_idx[1]][1])**2 ))
                    
                    self.tracks[min_idx[0]].position =  alpha * estimated_pos + (1-alpha) * np.array(keypoints[min_idx[1]][:2])
                    #self.tracks[min_idx[0]].position = np.array(keypoints[min_idx[1]][:2])
                    
                    tracks_matched[min_idx[0]] = 1 
                    matched_keypoints.append(min_idx[1])  

            # save updated track position
            for i in range(len(self.tracks)):
                self.tracks[i].history_estimated.append(self.tracks[i].position)

            # for each unmatched track: increase num_misses 
            for i,track in enumerate(self.tracks):
                if tracks_matched[i] == 0:
                    self.tracks[i].num_misses += 1 
                    if self.tracks[i].num_misses > self.max_num_misses: # lost track over so many consecutive frames
                        # delete later
                        if i not in delete_tracks:
                            delete_tracks.append(i)
                    if self.tracks[i].age < self.max_age_wo_detection: # miss already at young age are false positives
                        if i not in delete_tracks:
                            delete_tracks.append(i)
                else:
                    self.tracks[i].num_misses = 0
        return matched_keypoints, delete_tracks 

    def create_new_tracks(self, keypoints, matched_keypoints):
        for k in range(len(keypoints)):
            if not k in matched_keypoints:
                self.tracks.append(
                    KeypointTrack(len(self.tracks), keypoints[k][:2], keypoints[k][2]))

   

    def update(self, keypoints):
        """ 
            update internal track stats by newly detected keypoints 
            while not all tracks matched:
                find minimum distance between unmatched track and unmatched keypoint
                add keypoint to history of track    
        """
        
        matched_keypoints, delete_tracks = self.match_keypoints_and_update_tracks(keypoints)
        self.create_new_tracks(keypoints, matched_keypoints)
        
        if 0:
            # check if track has moved
            for i in range(len(self.tracks)):
                if len(self.tracks[i].history)>1 and np.linalg.norm( np.array(self.tracks[i].history[-2])- np.array(self.tracks[i].history[-1] )) < min_dist_movement:
                    self.tracks[i].num_wo_movement += 1
                else:
                    self.tracks[i].num_wo_movement = 0

                if self.tracks[i].num_wo_movement > max_steps_wo_movement:
                    if i not in delete_tracks:
                        delete_tracks.append(i)

        # update age of all tracks 
        for track in self.tracks:
            track.age += 1

        # delete tracks
        delete_tracks = sorted(delete_tracks)
        for i in delete_tracks[::-1]:
            self.graveyard.append(self.tracks[i])
            self.graveyard[-1].alive = False 
            del self.tracks[i]

        return self.tracks 