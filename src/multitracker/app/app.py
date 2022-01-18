"""
    Website for Handling Tracking
    Label Tool
        randomly pick unlabeled frame
        let user label it
        save it 

    python3.7 -m multitracker.app.app --model /home/alex/checkpoints/keypoint_tracking/Resocialization_Gruppe2_2-2020-06-09_17-50-12/trained_model.h5 --project_id 3
"""

from flask import Flask, render_template, Response, send_file, abort, redirect
from flask import request
from flask import jsonify
import os 
import io 
import json
from glob import glob 
from random import shuffle 
import subprocess
import numpy as np 
import cv2 as cv 

import pandas as pd 
import logging
from collections import deque

#from multitracker.be.db.dbconnection import get_connector
from multitracker.be import video, project
from multitracker.keypoint_detection import model
from multitracker.keypoint_detection import heatmap_drawing
from multitracker import util


app = Flask(__name__)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model',default=None)
    parser.add_argument('--project_id',type=int,default=None)
    parser.add_argument('--video_id',type=int,default=None)
    parser.add_argument('--upper_bound',type=int,default=4)
    
    parser.add_argument('--num_labeling_base',type=int,default=100)
    parser.add_argument('--open_gallery', dest='open_gallery', action='store_true')
    
    parser.add_argument('--trackannotation_skip', type=int, default = 15)
    parser.add_argument('--trackannotation_video', type=str, default = None)
    parser.add_argument('--data_dir', required=False, default = os.path.expanduser('~/data/multitracker'))
    args = parser.parse_args()
    os.environ['MULTITRACKER_DATA_DIR'] = args.data_dir
from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection(file_db=os.path.join(args.data_dir, 'data.db'))

count_active_steps = 0 

def get_frame_time(frame_idx):
    frame_idx = int(frame_idx)
    tmin = int(frame_idx/(30.*60.))
    tsec = int(frame_idx/30.-tmin*60.)
    return '%i:%imin'%(tmin,tsec)


video_reader = None 
loaded_video_id = None
skip_annotation_frames = None
frame_idx_trackannotation = None#1


frame = None 
#visual_mot = cv.MultiTracker_create()

tracker_create = [
    cv.TrackerCSRT_create,
    cv.TrackerBoosting_create,
    #cv.TrackerKDF_create,
    cv.TrackerGOTURN_create,
    #cv.TrackerHOG_create,
    cv.TrackerMedianFlow_create
][3]


last_user_boxes = []
for i in range(args.upper_bound):
    last_user_boxes.append([200*i,200*i,200*(i+1),200*(i+1)])

if args.video_id is not None:
    ## open video file
    # ignore first 5 frames
    #for _ in range(300):
    #    ret, frame = video_reader.read()
    #    frame_buffer.append(frame)
    #[Hframe,Wframe,_] = frame.shape
    total_frame_number = int(video_reader.get(cv.CAP_PROP_FRAME_COUNT))
    
tracking_results_cache = {}

def get_csv(csv_path):
    if not csv_path in tracking_results_cache:
        data = pd.read_csv(csv_path)
        data = data.apply(lambda col:pd.to_numeric(col, errors='coerce'))
        print('[*] read_csv', csv_path)
        print('headers',data.head())
        print(data[:20])
        
        tracking_results_cache[csv_path] = data 
    return tracking_results_cache[csv_path]

def get_trackannotation_path(project_id, video_id):
    return os.path.expanduser('~/data/multitracker/projects/%i/%i/trackannotation.csv' % (int(project_id), int(video_id)))


def log_interpolated_trackannotation(data):
    trackannotation_csv = get_trackannotation_path(data['project_id'],data['video_id'])
    if not os.path.exists(trackannotation_csv):
        # write header if not csv present
        with open(trackannotation_csv, 'w') as f:
            f.write('frame_idx,idv,x1,y1,x2,y2\n')
    
    for i in range(len(data['bboxes'])):
        data['bboxes'][i]['frame_idx'] = int(data['frame_idx'])

    # interpolate from last_user_boxes to new data
    global last_user_boxes
    interpolated_boxes = []
    #if int(data['frame_idx']) <= 1:
    
    if int(data['frame_idx']) > 1:
        for iskip in range(args.trackannotation_skip):
            alpha = (1+iskip)/float(args.trackannotation_skip)
            for i in range(args.upper_bound):
                #print(iskip,'BOX',i,data['bboxes'][i]['is_visible'],'')
                _d = {}
                _d['frame_idx'] = int(data['frame_idx'])+iskip - args.trackannotation_skip + 1 
                _d['id_ind'] = data['bboxes'][i]['id_ind']
                skip=False 
                for j,k in enumerate(['x1','y1','x2','y2']):
                    try:
                        _d[k] = (1.-alpha) * last_user_boxes[i][j] + alpha * data['bboxes'][i][k]
                    except: # happens if new object without history of last boxes
                        _d[k] = data['bboxes'][i][k]
                        skip=True
                #if skip and iskip == 0 and data['bboxes'][i]['is_visible']:
                #    interpolated_boxes.append( _d )
                #elif data['bboxes'][i]['is_visible']:
                if data['bboxes'][i]['is_visible'] and not skip:
                    interpolated_boxes.append( _d )

    if int(data['frame_idx']) < 2:
        for i in range(args.upper_bound):
            if data['bboxes'][i]['is_visible']:
                interpolated_boxes.append(data['bboxes'][i] )


    # write boxes to disk
    with open(trackannotation_csv, 'a+') as f:
        for bbox in interpolated_boxes:
            f.write('%s,%i,%f,%f,%f,%f\n' % ( bbox['frame_idx'],int(bbox['id_ind']),bbox['x1'],bbox['y1'],bbox['x2'],bbox['y2'] ))


@app.route('/get_next_annotation/<project_id>/<video_id>')
def get_next_annotation(project_id, video_id):
    project_id = int(project_id) 
    video_id = int(video_id) 
    
    labeled_frame_idxs_boundingboxes = db.get_labeled_bbox_frames(video_id)
    num_frames = 5000
    next_frame_idx = 5 + int(num_frames * len(labeled_frame_idxs_boundingboxes) / float(args.num_labeling_base) )
    return '<script>document.location.href = "/get_annotation/%i/%i/%i";</script>' % ( project_id, video_id, next_frame_idx)

@app.route('/get_next_trackannotation/<project_id>/<video_id>')
def get_next_trackannotation(project_id, video_id):
    project_id = int(project_id) 
    video_id = int(video_id) 
    
    global frame_idx_trackannotation
    if frame_idx_trackannotation is None: 
        # try to parse csv to continue
        trackannotation_csv = get_trackannotation_path(project_id,video_id)
        if os.path.isfile(trackannotation_csv):
            with open(trackannotation_csv, 'r') as f:
                lines = [ l.replace('\n','') for l in f.readlines() ]
                try:
                    frame_idx_trackannotation = int(lines[-1].split(',')[0]) # parse last frame idx 
                except:
                    frame_idx_trackannotation = 1 

    if frame_idx_trackannotation is None:
        frame_idx_trackannotation = 1 
    next_frame_idx = frame_idx_trackannotation
    return '<script>document.location.href = "/get_trackannotation/%i/%i/%i";</script>' % ( project_id, video_id, next_frame_idx)



@app.route('/get_annotation/<project_id>/<video_id>/<frame_id>')
def get_annotation(project_id, video_id, frame_id):
    return redirect_annotation_tool(project_id, video_id, frame_id, mode = "annotate")

@app.route('/get_trackannotation/<project_id>/<video_id>/<frame_id>')
def get_trackannotation(project_id, video_id, frame_id):
    return redirect_annotation_tool(project_id, video_id, frame_id, mode = "trackannotation")



def redirect_annotation_tool(project_id, video_id, frame_id, mode = "annotate"):
    project_id = int(project_id) 
    video_id = int(video_id) 
    frame_id = int(frame_id) 

    config = model.get_config(int(project_id))
    #if args.video_id is None:
    #    raise Exception("\n   please restart the app with argument --video_id and --upper_bound")

    global loaded_video_id
    global frame 
    global last_user_boxes

    #if loaded_video_id is None or not loaded_video_id == video_id:
    #    get_next_annotation_frame(project_id, video_id, int(np.random.uniform(1e6)))

    # load labeled frame idxs
    labeled_frame_idxs = db.get_labeled_frames(video_id)
    labeled_frame_idxs_boundingboxes = db.get_labeled_bbox_frames(video_id)
    
    num_db_frames = len(labeled_frame_idxs)
    # load annotation from database
    if mode == "annotate":
        animals = db.get_frame_annotations(video_id, frame_id )
    elif mode == "trackannotation":
       
        animals = None
        ## visual tracking
        ## let the user only trackannotation only every 5th frame, do frame-by-frame visual tracking in between
        last_frame = None 
        if frame is None:
            frame = get_next_annotation_frame(project_id, video_id, frame_id , should_send_file = False)
            # parse last user boxes from csv file 
            if os.path.isfile(get_trackannotation_path(project_id, video_id)):
                last_user_boxes = []
                with open(get_trackannotation_path(project_id, video_id),'r') as f:
                    lines = [l.replace('\n','') for l in f.readlines()][-int(args.upper_bound):]
                    for jj in range(len(lines)):
                        _frame_idx, _idv, _x1, _y1, _x2, _y2 = lines[jj].split(',')
                        last_user_boxes.append([float(_x1),float(_y1),float(_x2),float(_y2)])

        try:
            last_frame = np.array(frame, copy=True )
            scale = 700. / last_frame.shape[1]
        except:
            raise Exception('[*]   error with annotation tracks for track eval. have you given command line argument --trackannotation_video to the video you want to create track annotation for')
        # setup trackers
        visual_mot = cv.MultiTracker_create()
        for j, bbox in enumerate(last_user_boxes):
            # has format -> 'bboxes': [{'x1': 737.3437, 'y1': 137.385, 'x2': 937.343, 'y2': 337.385, 'id_ind': '1', 'db_id': None, 'is_visible': True}, ..]
            if len(bbox) >0:
                if last_frame is not None:
                    visual_mot.add(tracker_create(), cv.resize(last_frame, None, None, fx = scale, fy = scale), tuple([int(cc*scale) for cc in bbox]))

        # track for 5 frames
        for iskip in range(args.trackannotation_skip):
            frame = get_next_annotation_frame(project_id, video_id, frame_id + iskip , should_send_file = False)
            #vis = np.array(frame,copy=True); vis = cv.putText( vis, str(iskip), (20,20), cv.FONT_HERSHEY_COMPLEX, 0.75, [255,0,0], 2 )
            #cv.imshow('huhu',vis); cv.waitKey(0)
            (success, tracked_boxes) = visual_mot.update(cv.resize(frame, None, None, fx = scale, fy = scale)) 
            tracked_boxes = [ [ int(cc / scale) for cc in bbox ] for bbox in tracked_boxes ]
        
        animals = []
        _cnt_skip = 0
        # if no annotation around, init upper_bound many animals with keypoints
        for i in range(len(last_user_boxes)):
            bbox = last_user_boxes[i]
            is_visible = len(bbox) > 0 
            if len(bbox) > 0: # is visible!
                x1,y1,x2,y2 = tracked_boxes[i-_cnt_skip]
            else:
                x1,y1,x2,y2 = [200*i,200*i,200*(i+1),200*(i+1)]
                _cnt_skip += 1 

            keypoints = [ ]
            for j,kp_name in enumerate(config['keypoint_names']):
                keypoints.append({
                    'name': kp_name,
                    'x': x1 + 20*j,
                    'y': y1 + 23*j,
                    'db_id': None,
                    'is_visible': False 
                })
        
            animals.append({'id': str(i+1), 'box': [x1,y1,x2,y2], 'keypoints': keypoints, 'db_id': None, 'is_visible': is_visible})
    



    if animals is None:
        animals = []
        # if no annotation around, init upper_bound many animals with keypoints
        for i in range(int(args.upper_bound)):
            x1,y1,x2,y2 = [200*i,200*i,200*(i+1),200*(i+1)]
            keypoints = [ ]
            for j,kp_name in enumerate(config['keypoint_names']):
                keypoints.append({
                    'name': kp_name,
                    'x': x1 + 20*j,
                    'y': y1 + 23*j,
                    'db_id': None,
                    'is_visible': not mode == "trackannotation"
                })
            
            animals.append({'id': str(i+1), 'box': [x1,y1,x2,y2], 'keypoints': keypoints, 'db_id': None, 'is_visible': True})
    else:
        # parse db animals
        #for i, animal in enumerate(animals):
        #    print(i, animal)
        pass 

    animals_json = json.dumps(animals)
    return render_template('annotate.html', animals = animals, animals_json = animals_json, project_id = int(project_id), video_id = int(video_id), frame_idx = frame_id, keypoint_names = db.list_sep.join(config['keypoint_names']), sep = db.list_sep, num_db_frames = num_db_frames, labeling_mode = mode)

@app.route('/get_video_frame/<project_id>/<video_id>/<frame_id>')
def get_next_annotation_frame(project_id, video_id, frame_id, should_send_file = True):
    project_id = int(project_id) 
    video_id = int(video_id) 
    frame_id = int(frame_id) 
    

    global frame 
    global frame_idx 
    global loaded_video_id
    global video_reader
    global skip_annotation_frames
    if loaded_video_id is None or not loaded_video_id == video_id or frame_idx >= frame_id:
        frame_buffer = deque(maxlen=300)
        db.execute('select name, project_id from videos where id=%i;' % int(video_id))
        video_name, project_id = [ x for x in db.cur.fetchall()][0]
        video_filepath = os.path.join(args.data_dir, 'projects', str(project_id), 'videos', video_name)
        if args.trackannotation_video is not None:
            video_filepath = args.trackannotation_video
        if not os.path.exists(video_filepath):
            print('[* ERROR] could not find video', video_filepath)
        #csv_filepath = video_filepath.replace('.mp4','.csv')
        video_reader = cv.VideoCapture( video_filepath )
        total_frame_number = int(video_reader.get(cv.CAP_PROP_FRAME_COUNT))
        skip_annotation_frames = int(total_frame_number / args.num_labeling_base) -1 
        frame_idx = 0
        ret, frame = video_reader.read()
        
        loaded_video_id = video_id 
        print('[*] loaded video', video_filepath)
    
    while frame_idx < frame_id:
        ret, frame = video_reader.read()
        frame_idx += 1 
    if frame is None:
        return '''<html><body>you done annotating :) video over</body></html>'''
    

    if not should_send_file:
        return frame 
    is_success, im_buf_arr = cv.imencode(".jpg", frame)
    byte_im = im_buf_arr.tobytes()
    try:
        return send_file( io.BytesIO(byte_im),
                download_name='frame.jpg',
                mimetype='image/jpeg')  
    except: # flask < 2.0
        return send_file( io.BytesIO(byte_im),
            attachment_filename='frame.jpg',
            mimetype='image/jpeg')  
    #return (b'--frame\r\n'
    #           b'Content-Type: image/jpeg\r\n\r\n' + frame.tostring() + b'\r\n')




@app.route('/get_frame/<project_id>/<video_id>/<frame_idx>')
def get_frame(project_id,video_id,frame_idx):
    try:
        local_path = os.path.join(dbconnection.base_data_dir, "projects/%s/%s/frames/train/%s.png" % (project_id, video_id, frame_idx))
        return send_file(local_path, mimetype='image/%s' % local_path.split('.')[-1])
    except Exception as e:
        print('[E] /get_frame',project_id,video_id,frame_idx)
        print(e)
        return json.dumps({'success':False}), 200, {'ContentType':'application/json'} 

def gen_frame(project_id, video_id):
    project_id, video_id = int(project_id), int(video_id)
    data = inference.load_data(project_id, video_id)
    for i, file_name in enumerate(data):
        frame = cv.imread(file_name)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame.tostring() + b'\r\n')

@app.route('/get_video/<project_id>/<video_id>')
def get_video(project_id, video_id):
    local_path = '/home/dolokov/Downloads/Basler_127.mp4'
    project_id, video_id = int(project_id), int(video_id)
    #return send_file(local_path, mimetype='video/mp4')
    #return Response(open(local_path, "rb"), mimetype="video/mp4")
    return Response(gen_frame(project_id, video_id),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/refine_video/<project_id>/<video_id>')
def refine_video(project_id, video_id):
    project_name = db.get_project_name(int(project_id))
    video_name = db.get_video_name(int(video_id))
    return render_template('video_player.html', project_id=project_id, video_id=video_id, project_name = project_name, video_name= video_name)

@app.route('/label_later/<project_id>/<video_id>', methods = ["POST"])
def label_later(project_id, video_id):
    data = request.get_json(silent=True,force=True)
    timestamp = data['time']
    frame_name = '%05d.png' % int(timestamp * 30.)
    q = "insert into frame_jobs (project_id, video_id, time, frame_name) values (%i, %i, %f, '%s');" % (int(project_id),int(video_id),timestamp,frame_name)
    db.execute(q)
    db.commit()
    db.execute('select id, frame_name from frame_jobs where project_id=%i and video_id=%i;' % (int(project_id),int(video_id)))
    labeling_jobs = [ {'id':x[0],'frame_name':x[1]} for x in db.cur.fetchall()]
    
    print('[*] label later: %i jobs'%len(labeling_jobs), project_id, video_id, timestamp, 'frame', frame_name)
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/delete_labeling/<project_id>/<video_id>/<frame_idx>')
def delete_labeling(project_id, video_id, frame_idx):
    project_id = int(project_id)
    video_id = int(video_id)
    
    db.execute("delete from keypoint_positions where video_id=%i and frame_idx='%s'" % (video_id, frame_idx))
    db.commit()
    q = "insert into frame_jobs (project_id, video_id, frame_name) values (%i, %i, '%s');" % (int(project_id),int(video_id),frame_idx)
    db.execute(q)
    db.commit()
    
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

idx_checked = {} 
@app.route('/check_labeling/<project_id>/<video_id>')
def check_labeling_handler(project_id, video_id):
    """
        serve a little page to see the drawn labeling and two buttons for good and bad. bads should be deleted and added as a labeling job
    """
    project_id, video_id = int(project_id), int(video_id)
    config = model.get_config(project_id=project_id)
    
    if video_id not in idx_checked:
        idx_checked[video_id] = 0
    
    # first get all frames 
    q = "select frame_idx from keypoint_positions where video_id=%i order by id;" % video_id
    db.execute(q)
    frame_idxs = [x[0] for x in db.cur.fetchall()]
    frame_idxs = list(set(frame_idxs))
    frame_idx = frame_idxs[idx_checked[video_id]]

    q = "select keypoint_name, individual_id, keypoint_x, keypoint_y from keypoint_positions where video_id=%i and frame_idx='%s' order by individual_id, keypoint_name desc;" % (video_id,frame_idx)
    db.execute(q)
    keypoints = [x for x in db.cur.fetchall()]
    
    filepath = os.path.join(dbconnection.base_data_dir, "projects/%i/%i/frames/train/%s.png" % (int(project_id), int(video_id), frame_idx))
    im = cv.imread(filepath)
    
    hm = heatmap_drawing.generate_hm(im.shape[0], im.shape[1] , [ [int(kp[2]),int(kp[3]),kp[0]] for kp in keypoints ], config['keypoint_names'])
    colors = util.get_colors()
    vis = np.float32(im)
    hm_color = np.zeros_like(vis)
    for i in range(hm.shape[2]):
        hm8 = hm[:,:,i]
        hm8[hm8<0.5]=0
        c = np.array(colors[i]).reshape((1,1,3))
        c = np.tile(c,[hm.shape[0],hm.shape[1],1])
    
        hm8 = np.expand_dims(hm8,axis=2)
        hmc = c * np.tile(hm8,[1,1,3])
        hm_color += hmc 
    vis = np.uint8(vis/2. + hm_color/2.)
    vis_path = '/tmp/check_%i-%s.png' % (video_id, str(frame_idx)) 
    cv.imwrite(vis_path, vis)
    #print('[*] wrote',vis_path,vis.shape,vis.dtype,vis.min(),vis.max())
    idx_checked[video_id] += 1
    print('idx_checked',idx_checked[video_id])


    dom = """
        <html>
        <head>
            <link rel="stylesheet" type="text/css" href="https://semantic-ui.com/dist/semantic.min.css">
            <script
                src="https://code.jquery.com/jquery-3.1.1.min.js"
                integrity="sha256-hVVnYaiADRTO2PzUGmuLJr8BLUSjGIZsDYGmIJLv2b8="
                crossorigin="anonymous"></script>
            <script src="https://semantic-ui.com/dist/semantic.min.js"></script>
            <script src="/static/js/network.js"></script>
            <style>
                    * {
                        margin: 0;
                        padding: 0;
                    }
                    .imgbox {
                        height: 100%
                    }
                    
                </style>
        </head>
        <body>
            <div class="imgbox">
                <img height="80%" class="center-fit" src="/get_labeled_frame/[project_id]/[video_id]/[frame_idx]">
                
                <script type="text/javascript">
                    function onclick_ok(){ window.location.reload() }
                    function onclick_bad(){ 
                        get("/delete_labeling/[project_id]/[video_id]/[frame_idx]"); 
                        window.location.reload(); 
                    }
                </script>
                <button id="bu_ok" class="ui positive button" onclick="onclick_ok();">OK</button>
                <button id="bu_bad" class="ui negative button" onclick='onclick_bad();'>BAD</button>
            </div>
            
        </body>
        </html>
    """
    for _ in range(3):
        dom = dom.replace('[project_id]',str(project_id)).replace('[video_id]',str(video_id)).replace('[frame_idx]',str(frame_idx))
    return dom 
    # .imgbox {window.location.reload() margin: auto;                    }

@app.route('/get_labeled_frame/<project_id>/<video_id>/<frame_idx>')
def get_labeled_frame(project_id,video_id,frame_idx):
    try:
        # get labeling and draw it on image
        drawing_path = '/tmp/check_%i-%s.png' % (int(video_id), frame_idx) 
        print('[*] trying to read',drawing_path)
        #if os.path.isfile(drawing_path):
        #    os.remove(drawing_path)
        return send_file(drawing_path, mimetype='image/%s' % drawing_path.split('.')[-1])
    except Exception as e:
        print('[E] /get_frame',project_id,video_id,frame_idx)
        print(e)
        return json.dumps({'success':False}), 200, {'ContentType':'application/json'} 

@app.route('/home')
def get_home():
    return render_template("home.html")

@app.route('/get_videos')
def get_videos():
    db.execute('''select projects.id as project_id, projects.name as project_name, 
                                videos.name as video_name, videos.id as video_id, projects.manager, projects.keypoint_names
                                from projects 
                                inner join videos on projects.id = videos.project_id''')
    datalist = list(set([x for x in db.cur.fetchall()]))
    data = []
    counts_bbox = db.get_count_all_labeled_bbox_frames()
    counts_keypoints = db.get_count_all_labeled_frames()
    for i in range(len(datalist)):
        d = {}
        for ik, k in enumerate(['project_id','project_name','video_name','video_id','manager','keypoint_names']):
            d[k] = datalist[i][ik]
        d['keypoint_names'] = d['keypoint_names'].split(db.list_sep)
        d['video_name'] = d['video_name'].split('/')[-1]
        if d['video_id'] in counts_bbox:
            d['count_bboxes'] = counts_bbox[d['video_id']]
        else:
            d['count_bboxes'] = 0
        if d['video_id'] in counts_keypoints:
            d['count_keypoints'] = counts_keypoints[d['video_id']]
        else:
            d['count_keypoints'] = 0
        data.append(d)
    
    data = sorted(data, key = lambda x: (x['project_name'],x['video_name']))
    return json.dumps({'success':True,'data':data}), 200, {'ContentType':'application/json'} 

@app.route('/add_video', methods=["POST"])
def add_video():
    # read data 
    data = dict(request.form)
    print('[*] add project n video', data)
    
    # upload video
    uploaded_file = request.files['file']

    downloads_dir = os.path.join(dbconnection.base_data_dir,'downloads')
    if not os.path.isdir(downloads_dir): os.makedirs(downloads_dir)

    if uploaded_file.filename == '':
        return """<html><body><h1>You have not selected a file for upload! please add a new video again!</h1></body></html>"""

    local_video_file = os.path.join(downloads_dir, uploaded_file.filename)
    print('[*] file name upload', uploaded_file.filename)
    uploaded_file.save(local_video_file)

    # create project
    keypoint_names = data['keypoint-names'].replace('\n','').replace(' ','').split(',')

    project_id = project.create_project(data['project-name'], data['your-name'], keypoint_names)
    # create video
    fixed_number = data['fixed-number'].replace(' ','')
    try:
        fixed_number = int(fixed_number)
    except:
        fixed_number = 0
    video_id = video.add_video_to_project(video.base_dir_default, project_id, local_video_file, fixed_number)
    
    os.remove(local_video_file)
    return redirect('/home')
    

@app.route('/labeling',methods=["POST"])
def receive_labeling():
    data = request.get_json(silent=True,force=True)
    print('[*] received labeling for frame %i.'%(int(data['frame_idx']) ))
    #print(data)
    
    if data['labeling_mode'] in ['keypoint','annotate']:
        # save labeling to database
        db.save_keypoint_labeling(data)

    if data['labeling_mode'] in ['bbox','annotate']:
        db.save_bbox_labeling(data)
    
    if data['labeling_mode'] in ['trackannotation']:
        #print('[*] trackannotation mode. write', data)
        log_interpolated_trackannotation( data )
        
        global last_user_boxes
        last_user_boxes = []
        for bbox in data['bboxes']:
            if not 'is_visible' in bbox:
                print('\n[* WARNING] bbox has no "is_visible"', bbox)
            if 'is_visible' in bbox and bbox['is_visible']:
                last_user_boxes.append([bbox[k] for k in ['x1', 'y1', 'x2', 'y2'] ])
            else:
                last_user_boxes.append([])

        global frame_idx_trackannotation
        frame_idx_trackannotation += args.trackannotation_skip

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/skip_labeling',methods=["POST"])
def skip_labeling():
    data = request.get_json(silent=True,force=True)
    print('[*] skip labeling',data)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 




if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=8888)