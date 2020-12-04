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
import os 
import json
from glob import glob 
from random import shuffle 
import subprocess
import numpy as np 
import cv2 as cv 
import h5py 
import logging
import tensorflow as tf 
tf.get_logger().setLevel(logging.ERROR)

#from multitracker.be.db.dbconnection import get_connector
from multitracker.be import video, project
from multitracker.keypoint_detection import model, predict
from multitracker.keypoint_detection import heatmap_drawing
from multitracker import util
from multitracker.graph_tracking.__main__ import load_data

app = Flask(__name__)

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model',default=None)
    parser.add_argument('--project_id',type=int,default=None)
    parser.add_argument('--num_labeling_base',type=int,default=250)
    parser.add_argument('--open_gallery', dest='open_gallery', action='store_true')
    args = parser.parse_args()

# load neural network from disk (or init new one)
if args.model is not None and os.path.isfile(args.model):
    assert args.project_id is not None
    training_model = tf.keras.models.load_model(h5py.File(args.model, 'r'))
    print('[*] loaded model %s from disk' % args.model)
    config = model.get_config(project_id=args.project_id)
    optimizer = tf.keras.optimizers.Adam(config['kp_lr'])
    config = model.get_config(int(args.project_id))
else:
    training_model = None 
    

count_active_steps = 0 

def get_frame_time(frame_idx):
    frame_idx = int(frame_idx)
    tmin = int(frame_idx/(30.*60.))
    tsec = int(frame_idx/30.-tmin*60.)
    return '%i:%imin'%(tmin,tsec)

@app.route('/get_next_labeling_frame/<project_id>/<video_id>')
def render_labeling(project_id,video_id):
    config = model.get_config(int(project_id))
    video_id = int(video_id) 
    config['video_id'] = video_id
    
    # load labeled frame idxs
    labeled_frame_idxs = db.get_labeled_frames(video_id)
    labeled_frame_idxs_boundingboxes = db.get_labeled_bbox_frames(video_id)
    
    num_db_frames = len(labeled_frame_idxs)
    
    # load frame files from disk
    frames = []
    while len(frames) == 0:
        #video_id = db.get_random_project_video(project_id)
        frames_dir = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, project_id), video_id),'train')
        frames = sorted(glob(os.path.join(frames_dir, '*.png')))
    #print('[E] no frames found in directory %s.'%frames_dir)
    
    # look for unfinished labeling jobs 
    db.execute('select id, frame_name from frame_jobs where project_id=%i and video_id=%i;' % (int(project_id),int(video_id)))
    labeling_jobs = [ {'id':x[0],'frame_name':x[1]} for x in db.cur.fetchall()]
    if len(labeling_jobs) > 0:
        if not labeling_jobs[0]['frame_name'][:-4] == '.png':
            labeling_jobs[0]['frame_name'] += '.png'
        frame_idx = '.'.join(labeling_jobs[0]['frame_name'].split('.')[:-1])
        #print('A',labeling_jobs[0]['frame_name'],'->',frame_idx)
        print('[*] found %i labeling jobs, giving %i' % (len(labeling_jobs),labeling_jobs[0]['id']))

        # delete job
        db.execute('delete from frame_jobs where id=%i;' % labeling_jobs[0]['id'])
        db.commit()

    else:
        ## first check if labeled bounding box data with unlabeled keypoint data is in db
        unlabeled_labeled_boxes_frame_idx = []
        for frame_idx_bbox in labeled_frame_idxs_boundingboxes:
            if frame_idx_bbox not in labeled_frame_idxs:
                unlabeled_labeled_boxes_frame_idx.append(frame_idx_bbox)
        if len(unlabeled_labeled_boxes_frame_idx) > 0:
            print('[*] found %i labeled boxes frames, where no keypoints are annotated. here is one of them' % len(unlabeled_labeled_boxes_frame_idx))
            shuffle(unlabeled_labeled_boxes_frame_idx)
            frame_idx = unlabeled_labeled_boxes_frame_idx[0]
        else:
            if num_db_frames < args.num_labeling_base:
                '' # randomly sample frame 
                # choose random frame that is not already labeled
                unlabeled_frame_found = False 
                while not unlabeled_frame_found:
                    ridx = int( len(frames) * num_db_frames / float(args.num_labeling_base) ) + int(np.random.uniform(-5,5))
                    if ridx > 0 and ridx < len(frames):
                        frame_idx = frames[ridx]
                        #frame_idx = frames[int(len(frames)*np.random.random())] # random sampling
                        frame_idx = '.'.join(frame_idx.split('/')[-1].split('.')[:-1])
                        unlabeled_frame_found = not (frame_idx in labeled_frame_idxs)

                print('[*] serving keypoint label job for frame %s %s.'%(frame_idx,get_frame_time(frame_idx)))
            else:
                shuffle(frames)
                nn = 1#32
                unlabeled = []
                while len(unlabeled) < nn:
                    frame_f = frames[int(len(frames)*np.random.random())]
                    frame_idx = '.'.join(frame_f.split('/')[-1].split('.')[:-1])
                    nearest_labeled_frame_diff = np.min(np.abs(np.array([int(idx) for idx in labeled_frame_idxs]) - int(frame_idx)))
                    if frame_f not in unlabeled and frame_idx not in labeled_frame_idxs and nearest_labeled_frame_diff > 20:
                        unlabeled.append(frame_f)

                if training_model is not None:
                    # active learning: load some unlabeled frames, inference multiple time, take one with biggest std variation
                    
                    # predict whole image, height like trained height and variable width 
                    # to keep aspect ratio and relative size        
                    w = 1+int(2*config['img_height']/(float(cv.imread(unlabeled[0]).shape[0]) / cv.imread(unlabeled[0]).shape[1]))
                    batch = np.array( [cv.resize(cv.imread(f), (w,2*config['img_height'])) for f in unlabeled] ).astype(np.float32)
                    
                    # inference multiple times
                    ni = 5
                    pred = np.array([ training_model(batch,training=True)[-1].numpy() for _ in range(ni)])
                    pred = pred[:,:,:,:,:-1] # cut background channel
                    
                    # calculate max std item
                    stds = np.std(pred,axis=(0,2,3,4))
                    maxidx = np.argmax(stds)
                    frame_idx = '.'.join(unlabeled[maxidx].split('/')[-1].split('.')[:-1])
                    print('[*] serving keypoint label job for frame %s with std %f %s.'%(frame_idx,stds[maxidx],get_frame_time(frame_idx)))
                else:
                    if len(unlabeled) ==0:
                        return "<h1>You have labeled all frames for this video! :)</h1>"
                    shuffle(unlabeled)
                    frame_idx = '.'.join(unlabeled[0].split('/')[-1].split('.')[:-1])

    frame_candidates = []
    if 0:
        # inference to send frame candidates to client
        im = batch[maxidx]
        frame_candidates = [ ]
        imp = np.mean(pred[:,maxidx,:,:,:],axis=0)
        #print('im',im.shape,'imp',imp.shape,imp.min(),imp.max())
        for c in range(len(config['keypoint_names'])):
            frame_candidates.append(predict.extract_frame_candidates(imp[:,:,c],0.1))
        print('frame_candidates',frame_candidates)
        
    if len(labeled_frame_idxs) > 0:
        nearest_labeled_frame_diff = np.min(np.abs(np.array([int(idx) for idx in labeled_frame_idxs]) - int(frame_idx)))
    else:
        nearest_labeled_frame_diff = -1
    print('[*] serving keypoint label job for frame %s %s. nearest frame already labeled %i frames away'%(frame_idx,get_frame_time(frame_idx),nearest_labeled_frame_diff))       
    
    if args.open_gallery:
        p = subprocess.Popen(['eog',unlabeled[maxidx]])
    return render_template('labeling.html',project_id = int(project_id), video_id = int(video_id), frame_idx = frame_idx, keypoint_names = db.list_sep.join(config['keypoint_names']), sep = db.list_sep, num_db_frames = num_db_frames, frame_candidates = frame_candidates, labeling_mode = 'keypoint')


@app.route('/get_next_bbox_frame/<project_id>/<video_id>')
def get_next_bbox_frame(project_id, video_id):
    project_id = int(project_id)
    config = model.get_config(project_id)
    #video_id = db.get_random_project_video(project_id)
    video_id = int(video_id)
    config['video_id'] = video_id

    # load labeled frame idxs
    labeled_frames_keypoints = db.get_labeled_frames(video_id)
    labeled_frame_idxs = db.get_labeled_bbox_frames(video_id)
    num_db_frames = len(labeled_frame_idxs)
    
    # load frame files from disk
    frames_dir = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, project_id), video_id),'train')
    frames = sorted(glob(os.path.join(frames_dir, '*.png')))
    frames_keypoints = [os.path.join(frames_dir, '%s.png' % ff) for ff in labeled_frames_keypoints]
    shuffle(frames_keypoints)
    
    unlabeled_frame_found = False 
    if len(frames_keypoints) > 0:
        # first try to label bounding boxes where keypoints are labeled but bboxes not
        tries = 0
        while not unlabeled_frame_found and tries < 50:
            ridx = int(np.random.uniform(len(frames_keypoints)))
            frame_idx = frames_keypoints[ridx]
            frame_idx = '.'.join(frame_idx.split('/')[-1].split('.')[:-1])
            unlabeled_frame_found = not (frame_idx in labeled_frame_idxs)
            tries += 1 

    # then take random unlabeled frame
    tries, nearest_labeled_frame_diff = 0, 0
    while not unlabeled_frame_found and tries < 50:
        ridx = int(np.random.uniform(len(frames)))
        frame_idx = frames[ridx]
        frame_idx = '.'.join(frame_idx.split('/')[-1].split('.')[:-1])
        if len(labeled_frame_idxs) == 0: 
            unlabeled_frame_found = True 
        else:
            nearest_labeled_frame_diff = np.min(np.abs(np.array([int(idx) for idx in labeled_frame_idxs]) - int(frame_idx)))
            if nearest_labeled_frame_diff > 20:
                unlabeled_frame_found = not (frame_idx in labeled_frame_idxs)
        tries += 1 

    if len(labeled_frame_idxs) > 0:
        nearest_labeled_frame_diff = np.min(np.abs(np.array([int(idx) for idx in labeled_frame_idxs]) - int(frame_idx)))
    else:
        nearest_labeled_frame_diff = -1
    if unlabeled_frame_found:
        print('[*] serving bounding box label job for frame %s %s. nearest frame already labeled %i frames away'%(frame_idx,get_frame_time(frame_idx),nearest_labeled_frame_diff))       
        return render_template('labeling.html',project_id = int(project_id), video_id = int(video_id), frame_idx = frame_idx, num_db_frames = num_db_frames, keypoint_names = db.list_sep.join(config['keypoint_names']), sep = db.list_sep, labeling_mode = 'bbox')
    else:
        print('[*] redirecting to keypoint labeling')
        return render_labeling(project_id, video_id)

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
    data = load_data(project_id, video_id)
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
    
    if data['labeling_mode'] == 'keypoint':
        # save labeling to database
        db.save_labeling(data)

        if training_model is not None:
            # draw sent data
            frames_dir = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, data['project_id']), data['video_id']),'train')
            filepath = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, data['project_id']), data['video_id']),'train','%s.png'%data['frame_idx'])
            keypoints = [[d['keypoint_name'],d['id_ind'],d['x'],d['y']] for d in data['keypoints']]
            y = heatmap_drawing.vis_heatmap(cv.imread(filepath), int(data['frame_idx']) , config['keypoint_names'], keypoints, horistack = False)
            w = 1+int(2*config['img_height']/(float(y.shape[0]) / y.shape[1]))
            y = cv.resize(y,(w,2*config['img_height']))
            y = np.float32(y / 255.)
            bs = 1 # config['batch_size']
            y = np.expand_dims(y,axis=0)
            if bs > 1:
                y = np.tile(y,[bs,1,1,1])
            batch = np.tile(np.expand_dims(cv.resize(cv.imread(filepath), (w,config['img_height'])),axis=0),[bs,1,1,1])
            batch = batch.astype(np.float32)
            
            # train model with new data multiple steps
            ni = 3
            for i in range(ni):
                with tf.GradientTape(persistent=True) as tape:
                    predicted_heatmaps = training_model(batch, training=True)[-1]
                    y = y[:,:,:,:predicted_heatmaps.shape[3]]
                    predicted_heatmaps = tf.image.resize(predicted_heatmaps,(y.shape[1],y.shape[2]))
                    loss = 0.
                    for ph in predicted_heatmaps: 
                        loss += model.get_loss(ph, y, config)
                    print('loss',i,loss)
                # update network parameters
                gradients = tape.gradient(loss,training_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients,training_model.trainable_variables))

            global count_active_steps
            count_active_steps += 1
            # sometimes save model to disk
            if count_active_steps % 10 == 0:
                training_model.save(args.model)
    elif data['labeling_mode'] == 'bbox':
        #print(data)
        db.save_bbox_labeling(data)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/skip_labeling',methods=["POST"])
def skip_labeling():
    data = request.get_json(silent=True,force=True)
    print('[*] skip labeling',data)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 



if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=8888)