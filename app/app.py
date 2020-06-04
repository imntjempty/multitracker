"""
    Website for Handling Tracking
    Label Tool
        randomly pick unlabeled frame
        let user label it
        save it 
"""

from flask import Flask, render_template, Response, send_file, abort
from flask import request
import os 
import json
from glob import glob 
from random import shuffle 

import numpy as np 
import cv2 as cv 

#from multitracker.be.db.dbconnection import get_connector
from multitracker.be import video

app = Flask(__name__)

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

@app.route('/get_next_labeling_frame/<project_id>')
def render_labeling(project_id):
    video_id = db.get_random_project_video(project_id)
    
    # load labeled frame idxs
    labeled_frame_idxs = db.get_labeled_frames(project_id)

    # load frame files from disk
    frames_dir = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, project_id), video_id),'train')
    frames = sorted(glob(os.path.join(frames_dir, '*.png')))
    if len(frames) == 0:
        print('[E] no frames found in directory %s.'%frames_dir)
    shuffle(frames)

    # choose random frame that is not already labeled
    unlabeled_frame_found = False 
    while not unlabeled_frame_found:
        frame_idx = frames[int(len(frames)*np.random.random())]
        frame_idx = '.'.join(frame_idx.split('/')[-1].split('.')[:-1])
        unlabeled_frame_found = not (frame_idx in labeled_frame_idxs)
    num_db_frames = len(labeled_frame_idxs)
    
    keypoint_names = db.get_keypoint_names(project_id,split=False)
    return render_template('labeling.html',project_id = int(project_id), video_id = int(video_id), frame_idx = frame_idx, keypoint_names = keypoint_names, sep = db.list_sep, num_db_frames = num_db_frames)

@app.route('/get_frame/<project_id>/<video_id>/<frame_idx>')
def get_frame(project_id,video_id,frame_idx):
    try:
        #f = [dict(x) for x in get_connector()._execute("select s3_bucket,s3_png from scanfiles where id=%i;" % int(file_id))][0]
        #local_path = download_scanfile(s3c,download_dir,f)
        local_path = os.path.expanduser("~/data/multitracker/projects/%s/%s/frames/train/%s.png" % (project_id, video_id, frame_idx))
        return send_file(local_path, mimetype='image/%s' % local_path.split('.')[-1])
    except Exception as e:
        print('[E] /get_frame',project_id,video_id,frame_idx)
        print(e)
        return json.dumps({'success':False}), 200, {'ContentType':'application/json'} 


@app.route('/labeling',methods=["POST"])
def receive_labeling():
    data = request.get_json(silent=True,force=True)
    print('[*] received labeling for frame %i.'%(int(data['frame_idx']) ))
    
    # save labeling to database
    db.save_labeling(data)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/skip_labeling',methods=["POST"])
def skip_labeling():
    data = request.get_json(silent=True,force=True)
    print('[*] skip labeling',data)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)