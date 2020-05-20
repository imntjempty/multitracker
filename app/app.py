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

import numpy as np 
import cv2 as cv 

#from multitracker.be.db.dbconnection import get_connector


app = Flask(__name__)



@app.route('/get_next_labeling_frame/<project_id>')
def render_labeling(project_id):
    video_id = 0
    return render_template('labeling.html',project_id = int(project_id), video_id = int(video_id), frame_idx = frame_idx)

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
    print(data)
    print()

@app.route('/skip_labeling',methods=["POST"])
def skip_labeling():
    data = request.get_json(silent=True,force=True)
    print('[*] skip labeling',data)