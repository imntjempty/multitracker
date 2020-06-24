"""
    Website for Handling Tracking
    Label Tool
        randomly pick unlabeled frame
        let user label it
        save it 

    python3.7 -m multitracker.app.app --model /home/alex/checkpoints/keypoint_tracking/Resocialization_Gruppe2_2-2020-06-09_17-50-12/trained_model.h5 --project_id 3
"""

from flask import Flask, render_template, Response, send_file, abort
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
from multitracker.be import video
from multitracker.keypoint_detection import model, predict
from multitracker.keypoint_detection import heatmap_drawing


app = Flask(__name__)

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model',default=None)
    parser.add_argument('--project_id',type=int,default=None)
    parser.add_argument('--num_labeling_base',type=int,default=50)
    parser.add_argument('--open_gallery', dest='open_gallery', action='store_true')
    args = parser.parse_args()

# load neural network from disk (or init new one)
if args.model is not None and os.path.isfile(args.model):
    assert args.project_id is not None
    training_model = tf.keras.models.load_model(h5py.File(args.model, 'r'))
    print('[*] loaded model %s from disk' % args.model)
    optimizer = tf.keras.optimizers.Adam(config['lr'])
    config = model.get_config(int(args.project_id))
else:
    training_model = None 
    '''if args.model is None:
        args.model = '/tmp/active_model_%i.h5' % int(args.project_id)
    from multitracker.keypoint_detection import nets
    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 3])
    encoder = nets.Encoder(config,inputs)
    training_model = nets.Decoder(config,encoder)
    print('[*] creating new model %s from scratch' % args.model)'''
    

count_active_steps = 0 

@app.route('/get_next_labeling_frame/<project_id>')
def render_labeling(project_id):
    config = model.get_config(int(project_id))
    
    # load labeled frame idxs
    labeled_frame_idxs = db.get_labeled_frames(project_id)
    num_db_frames = len(labeled_frame_idxs)
    
    # load frame files from disk
    frames = []
    while len(frames) == 0:
        video_id = db.get_random_project_video(project_id)
        frames_dir = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, project_id), video_id),'train')
        frames = sorted(glob(os.path.join(frames_dir, '*.png')))
    #print('[E] no frames found in directory %s.'%frames_dir)
    

    if num_db_frames < args.num_labeling_base:
        '' # randomly sample frame 
        # choose random frame that is not already labeled
        unlabeled_frame_found = False 
        while not unlabeled_frame_found:
            ridx = int( len(frames) * num_db_frames / float(args.num_labeling_base) ) + int(np.random.uniform(-50,50))
            if ridx > 0 and ridx < len(frames):
                frame_idx = frames[ridx]
                #frame_idx = frames[int(len(frames)*np.random.random())] # random sampling
                frame_idx = '.'.join(frame_idx.split('/')[-1].split('.')[:-1])
                unlabeled_frame_found = not (frame_idx in labeled_frame_idxs)

        print('[*] serving label job for frame %s.'%(frame_idx))
    else:
        shuffle(frames)
        nn = 1#32
        unlabeled = []
        while len(unlabeled) < nn:
            frame_f = frames[int(len(frames)*np.random.random())]
            frame_idx = '.'.join(frame_f.split('/')[-1].split('.')[:-1])
            if frame_f not in unlabeled and frame_idx not in labeled_frame_idxs:
                unlabeled.append(frame_f)
        if training_model is not None:
            '' # active learning: load some unlabeled frames, inference multiple time, take one with biggest std variation
            
            # predict whole image, height like trained height and variable width 
            # to keep aspect ratio and relative size        
            w = 1+int(config['img_height']/(float(cv.imread(unlabeled[0]).shape[0]) / cv.imread(unlabeled[0]).shape[1]))
            batch = np.array( [cv.resize(cv.imread(f), (w,config['img_height'])) for f in unlabeled] ).astype(np.float32)
            
            # inference multiple times
            ni = 5
            pred = np.array([ training_model(batch,training=True).numpy() for _ in range(ni)])
            pred = pred[:,:,:,:,:-1] # cut background channel
            
            # calculate max std item
            stds = np.std(pred,axis=(0,2,3,4))
            maxidx = np.argmax(stds)
            frame_idx = '.'.join(unlabeled[maxidx].split('/')[-1].split('.')[:-1])
            print('[*] serving label job for frame %s with std %f.'%(frame_idx,stds[maxidx]))

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
        
            
    
    if args.open_gallery:
        p = subprocess.Popen(['eog',unlabeled[maxidx]])
    return render_template('labeling.html',project_id = int(project_id), video_id = int(video_id), frame_idx = frame_idx, keypoint_names = db.list_sep.join(config['keypoint_names']), sep = db.list_sep, num_db_frames = num_db_frames, frame_candidates = frame_candidates)

    

@app.route('/get_frame/<project_id>/<video_id>/<frame_idx>')
def get_frame(project_id,video_id,frame_idx):
    try:
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

    if training_model is not None:
        # draw sent data
        filepath = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, data['project_id']), data['video_id']),'train','%s.png'%data['frame_idx'])
        keypoints = [[d['keypoint_name'],d['id_ind'],d['x'],d['y']] for d in data['keypoints']]
        y = heatmap_drawing.vis_heatmap(cv.imread(filepath), config['keypoint_names'], keypoints, horistack = False)
        w = 1+int(config['img_height']/(float(y.shape[0]) / y.shape[1]))
        y = cv.resize(y,(w,config['img_height']))
        y = np.float32(y / 255.)
        
        y = np.tile(np.expand_dims(y,axis=0),[config['batch_size'],1,1,1])
        batch = np.tile(np.expand_dims(cv.resize(cv.imread(filepath), (w,config['img_height'])),axis=0),[config['batch_size'],1,1,1])
        batch = batch.astype(np.float32)
        
        # train model with new data multiple steps
        ni = 3
        for i in range(ni):
            with tf.GradientTape(persistent=True) as tape:
                predicted_heatmaps = training_model(batch, training=True)
                y = y[:,:,:,:predicted_heatmaps.shape[3]]
                predicted_heatmaps = tf.image.resize(predicted_heatmaps,(y.shape[1],y.shape[2]))
                loss = model.get_loss(predicted_heatmaps, y, config)
        
            # update network parameters
            gradients = tape.gradient(loss,training_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,training_model.trainable_variables))

        global count_active_steps
        count_active_steps += 1
        # sometimes save model to disk
        if count_active_steps % 10 == 0:
            training_model.save(args.model)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/skip_labeling',methods=["POST"])
def skip_labeling():
    data = request.get_json(silent=True,force=True)
    print('[*] skip labeling',data)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 



if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=8888)