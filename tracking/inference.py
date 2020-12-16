
import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py
import multiprocessing as mp 

from multitracker import util 
from multitracker.keypoint_detection import heatmap_drawing, model 
from multitracker.keypoint_detection.blurpool import BlurPool2D
from multitracker import autoencoder
from multitracker.tracking.deep_sort.deep_sort.detection import Detection
from multitracker.keypoint_detection import roi_segm, unet
from multitracker.be import video

from multitracker.be import dbconnection

def get_project_frame_test_dir(project_id, video_id):
    return os.path.join( dbconnection.base_data_dir, 'projects/%i/%i/frames/test' % (project_id,video_id))
def get_project_frame_train_dir(project_id, video_id):
    return os.path.join(dbconnection.base_data_dir, 'projects/%i/%i/frames/train' % (project_id,video_id))

def extract_frame_candidates(feature_map, thresh = 0.75, pp = 5):
    step = -1
    max_step = 50
    stop_threshold_hit = False 
    frame_candidates = []
    while not stop_threshold_hit and step < max_step:
        step += 1
        # find new max pos
        max_pos = np.unravel_index(np.argmax(feature_map),feature_map.shape)
        py = max_pos[0] #max_pos // feature_map.shape[0]
        px = max_pos[1] #max_pos % feature_map.shape[1]
        val = feature_map[py][px] #np.max(feature_map)
        frame_candidates.append([px,py,val])

        # delete area around new max pos 
        feature_map[py-pp:py+pp,px-pp:px+pp] = 0
        feature_map[py][px] = 0 
        
        # stop extraction if max value has small probability 
        if val < thresh:
            frame_candidates = frame_candidates[:-1]
            stop_threshold_hit = True 
    return frame_candidates

def get_video_output_filepath(config):
    if not 'kp_num_hourglass' in config:
        config['kp_num_hourglass'] = 1 
    if 'video' in config and config['video'] is not None:
        video_file_out = os.path.join(video.get_project_dir(video.base_dir_default, config['project_id']),'tracking_%s_%s_%s_%istack_%s_%s.avi' % (config['project_name'],config['tracking_method'],config['object_detection_backbone'],config['kp_num_hourglass'], config['kp_backbone'],'.'.join(config['video'].split('/')[-1].split('.')[:-1])))
    else:
        video_file_out = os.path.join(video.get_project_dir(video.base_dir_default, config['project_id']),'tracking_%s_%s_%s_%istack_%s_vid%i.avi' % (config['project_name'],config['tracking_method'],config['object_detection_backbone'],config['kp_num_hourglass'], config['kp_backbone'],config['video_id']))
    return video_file_out

def load_keypoint_model(path_model):
    t0 = time.time()
    trained_model = tf.keras.models.load_model(h5py.File(os.path.join(path_model,'trained_model.h5'), 'r'),custom_objects={'BlurPool2D':BlurPool2D})
    t1 = time.time()
    print('[*] loaded keypoint model from %s in %f seconds.' %(path_model,t1-t0))
    return trained_model 

def load_data(project_id,video_id,max_minutes=0):
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

def get_heatmaps_keypoints(heatmaps, thresh_detection=0.5):
    x = np.array(heatmaps,copy=True)
    keypoints = [] 
    for c in range(x.shape[2]-1): # dont extract from background channel
        channel_candidates = extract_frame_candidates(x[:,:,c], thresh = thresh_detection, pp = int(0.02 * np.min(x.shape[:2])))
        for [px,py,val] in channel_candidates:
            keypoints.append([px,py,c])

    # [debug] filter for nose 
    #keypoin1s = [kp for kp in keypoints if kp[2]==1]
    return keypoints 

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

def inference_keypoints(config, frame, detections, keypoint_model, crop_dim, min_confidence_keypoints):
    if len(detections) == 0:
            keypoints = []
    else:
        # inference keypoints for all detections
        y_kpheatmaps = np.zeros((frame.shape[0],frame.shape[1],1+len(config['keypoint_names'])),np.float32)
        rois, centers = [],[]
        frame_kp = unet.preprocess(config, frame)
        #for i, track in enumerate(tracker.tracks):
        for i, detection in enumerate(detections):
            x1,y1,x2,y2 = detection.to_tlbr()
            # crop region around center of bounding box
            center = roi_segm.get_center(x1,y1,x2,y2, frame.shape[0], frame.shape[1], crop_dim)
            center[0] = int(round(center[0]))
            center[1] = int(round(center[1]))
            centers.append(center)
            roi = frame_kp[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:]
            roi = tf.image.resize(roi,[224,224])
            #roi = tf.expand_dims(tf.convert_to_tensor(roi),axis=0)
            rois.append(roi)
        rois = tf.stack(rois,axis=0)
        yroi = keypoint_model(rois, training=False)[-1]
        #yroi = yroi[0,:,:,:]
        yroi = tf.image.resize(yroi,(crop_dim//2*2,crop_dim//2*2)).numpy()
        for i in range(len(detections)):
            y_kpheatmaps[centers[i][0]-crop_dim//2:centers[i][0]+crop_dim//2,centers[i][1]-crop_dim//2:centers[i][1]+crop_dim//2,:] = yroi[i,:,:,:]
        if 0:
            for ik, keypoint_name in enumerate(config['keypoint_names']):
                print('HM',ik,keypoint_name,'minmax',y_kpheatmaps[:,:,ik].min(),y_kpheatmaps[:,:,ik].max(),'meanstd',y_kpheatmaps[:,:,ik].mean(),y_kpheatmaps[:,:,ik].std())
        keypoints = get_heatmaps_keypoints(y_kpheatmaps, thresh_detection=min_confidence_keypoints)
    return keypoints


# Again, uncomment this decorator if you want to run inference eagerly
#@tf.function
def detect_bounding_boxes(detection_model, input_tensor):
    """Run detection on an input image.

    Args:
        input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.

    Returns:
        A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
        and `detection_scores`).
    """
    #if len(inp_tensor.shape)==3:
    #    input_tensor = tf.expand_dims(input_tensor, 0)
    input_tensor = tf.cast(input_tensor,tf.float32)
    #print('SHAPES',input_tensor.shape)
    shapes = tf.constant(1 * [[640, 640, 3]], dtype=tf.int32)
    #input_tensor = tf.expand_dims(input_tensor,axis=0)
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    #preprocessed_image = tf.image.resize(preprocessed_image,(640,640))
    #print('preprocessed_image',preprocessed_image.shape,'shapes',shapes)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    detections['detection_boxes'] = detections['detection_boxes'][0,:,:]
    detections['detection_scores'] = detections['detection_scores'][0]
    return detections

def _downscale_mp(im):
    scale = 650. / min(im.shape[:2])
    return cv.resize(im,None,None,fx=scale,fy=scale)
    
#@tf.function
def detect_batch_bounding_boxes(detection_model, frames, seq_info, thresh_detection):
    scaled = []
    with mp.Pool(processes=os.cpu_count()) as pool:
        detresults = []
        for i in range(frames.shape[0]):
            detresults.append(pool.apply_async(_downscale_mp,(frames[i,:,:,:],)))
        for result in detresults:
            scaled.append(result.get())
    scaled = np.stack(scaled,axis=0)
    preprocessed_image, shapes = detection_model.preprocess(tf.convert_to_tensor(scaled))
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    bboxes = detection_model.postprocess(prediction_dict, shapes)
    results = []
    for b in range(bboxes['detection_boxes'].shape[0]):
        result = []
        for j in range(bboxes['detection_boxes'][b].shape[0]):
            class_id = 1
            proba = bboxes['detection_scores'][b][j]
            if proba > thresh_detection:
                #print('box',j,bboxes['detection_boxes'][j])
                top,left,height,width = bboxes['detection_boxes'][b][j]
                top *= frames.shape[1] #seq_info['image_size'][0]#/scale
                height *= frames.shape[1] #seq_info['image_size'][0]#/scale
                left *= frames.shape[2] #seq_info['image_size'][1]#/scale
                width *= frames.shape[2] #seq_info['image_size'][1]#/scale
                height = height - top  
                width = width - left

                detection = Detection([left,top,width,height], proba, np.zeros((1)))
                
                result.append(detection)  
        results.append(result)
    return results


def detect_frame_boundingboxes(config, detection_model, encoder_model, seq_info, frame, frame_idx, thresh_detection = 0.3):
    inp_tensor = frame #cv.imread(frame_file)
    H,W = inp_tensor.shape[:2]
    inp_tensor = cv.resize(inp_tensor,(640,640))
    inp_tensor = tf.expand_dims(inp_tensor,0)
    features, _ = encoder_model(autoencoder.preprocess(inp_tensor),training=False)
    features = tf.image.resize(features,[H,W])
    
    bboxes = detect_bounding_boxes(detection_model, inp_tensor)
    #for i in range(bboxes['detection_boxes'].shape[0]):
    results = []
    for j in range(bboxes['detection_boxes'].shape[0]):
        class_id = 1
        proba = bboxes['detection_scores'][j]
        if proba > thresh_detection:
            #print('box',j,bboxes['detection_boxes'][j])
            top,left,height,width = bboxes['detection_boxes'][j]
            top *= seq_info['image_size'][0]
            height *= seq_info['image_size'][0]
            left *= seq_info['image_size'][1]
            width *= seq_info['image_size'][1]
            height = height - top  
            width = width - left

            itop,iheight,ileft,iwidth = [int(ii) for ii in [top,height,left,width]]
            features_crop = features[:,itop:itop+iheight,ileft:ileft+iwidth,:]
            features_crop = tf.image.resize(features_crop,[64,64])
            features_crop = tf.keras.layers.Flatten()(features_crop)

            features_crop = features_crop.numpy()[0,:]
            # normalize to unit length
            features_crop = features_crop / np.linalg.norm(features_crop)

            detection = Detection([left,top,width,height], proba, features_crop)
            
            results.append(detection)        

    return inp_tensor[0,:,:,:], results        


def load_autoencoder_feature_extractor(config):
    ## feature extractor
    config.update({'img_height':640, 'img_width': 640})
    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 3])
    feature_extractor,encoder = autoencoder.Encoder(inputs)
    encoder_model = tf.keras.Model(inputs = inputs, outputs = [feature_extractor,encoder])
    ckpt = tf.train.Checkpoint(encoder_model=encoder_model)

    ckpt_manager = tf.train.CheckpointManager(ckpt, config['autoencoder_model'], max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)
    else:
        print('[*] WARNING: could not load pretrained model!')
    return encoder_model