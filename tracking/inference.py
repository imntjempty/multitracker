
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
from collections import deque
import torch

from multitracker import util 
from multitracker.be import video
from multitracker.be import dbconnection
from multitracker import autoencoder
from multitracker.keypoint_detection import heatmap_drawing, model 
from multitracker.keypoint_detection.blurpool import BlurPool2D
from multitracker.keypoint_detection.nets import ReflectionPadding2D
from multitracker.keypoint_detection import roi_segm, unet
from multitracker.tracking.deep_sort.deep_sort.detection import Detection

from multitracker.object_detection.YOLOX.tools import demo 
from multitracker.object_detection.YOLOX.yolox.exp import get_exp
from multitracker.object_detection.YOLOX.yolox.data.data_augment import ValTransform

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
        video_file_out = os.path.join(video.get_project_dir(video.base_dir_default, config['project_id']), 'tracking_%s_%s_%s_%istack_%s_%s.avi' % (config['project_name'],config['tracking_method'],config['object_detection_backbone'],config['kp_num_hourglass'], config['kp_backbone'],'.'.join(config['video'].split('/')[-1].split('.')[:-1])))
    else:
        video_file_out = os.path.join(video.get_project_dir(video.base_dir_default, config['project_id']), 'tracking_%s_%s_%s_%istack_%s_vid%i.avi' % (config['project_name'],config['tracking_method'],config['object_detection_backbone'],config['kp_num_hourglass'], config['kp_backbone'],config['test_video_ids'].split(',')[0]))
    return video_file_out

def load_keypoint_model(path_model):
    t0 = time.time()
    trained_model = tf.keras.models.load_model(h5py.File(os.path.join(path_model,'trained_model.h5'), 'r'),custom_objects={'BlurPool2D':BlurPool2D, 'ReflectionPadding2D': ReflectionPadding2D})
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
            roi = tf.image.resize(roi,[config['img_height'],config['img_width']])
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

def inference_batch_keypoints(config, keypoint_model, crop_dim, frames_tensor, detection_buffer, min_confidence_keypoints):
    # crop all the detections from all the frames and stack them batches on which you run inference. 
    # be careful to keep track of how many kps per frame, because we need to resort them afterwards
    keypoints_per_frame = deque()

    rois = []
    centers = [] 
    lens = []
    y_kpheatmaps = {}
    for j in range(frames_tensor.shape[0]):
        frame = frames_tensor[j,:,:,:]
        detections = detection_buffer[j]
        lens.append(len(detections))
        for i, detection in enumerate(detections):
            x1,y1,x2,y2 = detection.to_tlbr()
            # crop region around center of bounding box
            center = roi_segm.get_center(x1,y1,x2,y2, frames_tensor.shape[1], frames_tensor.shape[2], crop_dim)
            center[0] = int(round(center[0]))
            center[1] = int(round(center[1]))
            centers.append(center)
            roi = frame[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:]
            roi = tf.image.resize(roi,[config['img_height'],config['img_width']])
            rois.append(roi)
        
        if (len(rois) >= len(detection_buffer) or j == frames_tensor.shape[0]-1):
            rois = tf.stack(rois,axis=0)
            #print('   rois',j,rois.shape,frames_tensor.shape,'len(detection_buffer)',len(detection_buffer),'len(detections)',len(detections))
            if rois.shape[0] > 0:
                yroi = keypoint_model(rois, training=False)
                if len(yroi[-1].shape) == 4:
                    yroi = yroi[-1]

                yroi = tf.image.resize(yroi,(crop_dim//2*2,crop_dim//2*2)).numpy()
                while len(lens)>0:
                    y_kpheatmaps = np.zeros((frame.shape[0],frame.shape[1],1+len(config['keypoint_names'])),np.float32)
                    for kd in range(lens[0]):
                        y_kpheatmaps[centers[kd][0]-crop_dim//2:centers[kd][0]+crop_dim//2,centers[kd][1]-crop_dim//2:centers[kd][1]+crop_dim//2,:] = yroi[kd,:,:,:]
                    yroi = yroi[lens[0]:,:,:,:]
                    centers = centers[lens[0]:]
                    keypoints = get_heatmaps_keypoints(y_kpheatmaps, thresh_detection=min_confidence_keypoints)
                    keypoints_per_frame.append(keypoints)
                    lens = lens[1:]
            else:
                while len(lens)>0:
                    keypoints_per_frame.append([])
                    lens = lens[1:]
            lens=[]
            rois = []
            center = []
        
        
    return keypoints_per_frame

# Again, uncomment this decorator if you want to run inference eagerly
#@tf.function
def detect_bounding_boxes(config, detection_model, input_tensor):
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
    shapes = tf.constant(1 * [[config['object_detection_resolution'][1], config['object_detection_resolution'][0], 3]], dtype=tf.int32)
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    detections['detection_boxes'] = detections['detection_boxes'][0,:,:]
    detections['detection_scores'] = detections['detection_scores'][0]
    return detections

#@tf.function
def detect_batch_bounding_boxes_tf2(config, detection_model, frames, thresh_detection, encoder_model = None):
    scaled = []
    for i in range(frames.shape[0]):
        scaled.append(cv.resize( frames[i,:,:,:] ,(config['object_detection_resolution'][0],config['object_detection_resolution'][1])))
        
    scaled = np.stack(scaled,axis=0)
    preprocessed_image, shapes = detection_model.preprocess(tf.convert_to_tensor(scaled))
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    bboxes = detection_model.postprocess(prediction_dict, shapes)
    features = np.zeros((bboxes['detection_boxes'].shape[0],4))
    if encoder_model is not None:
        ae_config = autoencoder.get_autoencoder_config()
        ae_scaled = []
        for i in range(frames.shape[0]):
            ae_scaled.append(cv.resize( frames[i,:,:,:] ,(ae_config['ae_resolution'][0],ae_config['ae_resolution'][1])))
        ae_scaled = np.stack(ae_scaled,axis=0)
        ae_scaled = (ae_scaled / 127.5) - 1 # [0,255] => [-1,1]
        features = encoder_model( ae_scaled )[0]
        features = tf.keras.layers.GlobalAveragePooling2D()(features).numpy()
        #features = (features - features.mean())/features.std()
        features = features / np.linalg.norm(features)
        #print('features',features.shape,features.min(),features.max(),'meanstd',features.mean(),features.std())
        
    results = []
    for b in range(bboxes['detection_boxes'].shape[0]):
        result = []
        for j in range(bboxes['detection_boxes'][b].shape[0]):
            class_id = 1
            proba = bboxes['detection_scores'][b][j]
            if proba > thresh_detection:
                #print('box',j,bboxes['detection_boxes'][j])
                top, left, height, width = bboxes['detection_boxes'][b][j]
                top *= frames.shape[1]
                height *= frames.shape[1] 
                left *= frames.shape[2]
                width *= frames.shape[2] 
                height = height - top  
                width = width - left
    
                detection = Detection([left,top,width,height], proba, features[b,:])
                
                result.append(detection)  
        results.append(result)
    
    return results

def detect_batch_bounding_boxes(config, detection_model, frames, thresh_detection, encoder_model = None):
        
    frames = np.stack(frames,axis=0)
    ori_shape = frames.shape
    print('frames',frames.shape)
    
    num_classes = 1 
    nmsthre = 0.5
    batch_detections = []
    for i in range(frames.shape[0]):
        img = frames[i,:,:,:]
        img, _ = ValTransform(legacy=False)(img, None, config['object_detection_resolution'])
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if 1:#self.device == "gpu":
            img = img.cuda()
            if 1:#self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = detection_model(img)
            #if self.decoder is not None:
            #    outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = demo.postprocess(
                outputs, num_classes, thresh_detection,
                nmsthre, class_agnostic=True
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        ok = True
        try:
            outputs = outputs[0].cpu().numpy()
        except:
            batch_detections.append([])
            ok = False 
        
        if ok:
            if len(outputs) == 0:
                batch_detections.append([])
            else:
                bboxes = outputs[:, 0:4]

                # preprocessing: resize
                ratio = min(config['object_detection_resolution'][0] / ori_shape[1], config['object_detection_resolution'][1] / ori_shape[2])
                bboxes /= ratio
                
                cls = outputs[:, 6]
                scores = outputs[:, 4] * outputs[:, 5]

            
                features = np.zeros((len(bboxes),4))
                '''if encoder_model is not None:
                    ae_config = autoencoder.get_autoencoder_config()
                    ae_scaled = []
                    for i in range(frames.shape[0]):
                        ae_scaled.append(cv.resize( frames[i,:,:,:] ,(ae_config['ae_resolution'][0],ae_config['ae_resolution'][1])))
                    ae_scaled = np.stack(ae_scaled,axis=0)
                    ae_scaled = (ae_scaled / 127.5) - 1 # [0,255] => [-1,1]
                    features = encoder_model( ae_scaled )[0]
                    features = tf.keras.layers.GlobalAveragePooling2D()(features).numpy()
                    #features = (features - features.mean())/features.std()
                    features = features / np.linalg.norm(features)'''

                result = []
                for j, (bbox, class_name, score) in enumerate(zip(bboxes, cls, scores)):
                    left,top,x2,y2 = bbox
                    width, height = x2 - left, y2-top
                    #centerx, centery, width, height = bbox
                    #left = centerx - width/2.
                    #top = centery - height/2.
                    #print(i, left,top,width,height, score, 'features', features.shape)
                    detection = Detection([left,top,width,height], score, features[j,:])
                    
                    result.append(detection) 
                batch_detections.append(result) 

    return batch_detections 

def detect_frame_boundingboxes(config, detection_model, encoder_model, frame, frame_idx, thresh_detection = 0.3):
    inp_tensor = frame #cv.imread(frame_file)
    H,W = inp_tensor.shape[:2]
    inp_tensor = cv.resize(inp_tensor,(config['object_detection_resolution'][0],config['object_detection_resolution'][1]))
    inp_tensor = tf.expand_dims(inp_tensor,0)
    features, _ = encoder_model(autoencoder.preprocess(inp_tensor),training=False)
    features = tf.image.resize(features,[H,W])
    
    bboxes = detect_bounding_boxes(config, detection_model, inp_tensor)
    results = []
    for j in range(bboxes['detection_boxes'].shape[0]):
        class_id = 1
        proba = bboxes['detection_scores'][j]
        if proba > thresh_detection:
            #print('box',j,bboxes['detection_boxes'][j])
            top,left,height,width = bboxes['detection_boxes'][j]
            top *= frame.shape[0]
            height *= frame.shape[0]
            left *= frame.shape[1]
            width *= frame.shape[1]
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
    config_autoencoder = autoencoder.get_autoencoder_config()
    inputs = tf.keras.layers.Input(shape=[config_autoencoder['img_height'], config_autoencoder['img_width'], 3])
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

def load_object_detector(args):
    

    exp = get_exp(args['yolox_exp'], args['yolox_name'])

    #if args.conf is not None:
    #    exp.test_conf = args.conf
    #if args.nms is not None:
    exp.nmsthre = args['min_confidence_boxes']
    #if args.tsize is not None:
    #    exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    #logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    ckpt = torch.load(args['objectdetection_model'], map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    
    fp16 = True
    if 1:#args.device == "gpu":
        model.cuda()
        if fp16:
            model.half()  # to FP16
    model.eval()
    
    '''if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")'''

    return model 
    pred = demo.Predictor(
        model,
        exp,
        cls_names=['animal'],
        trt_file=None,
        decoder=None,
        device="gpu",
        fp16=False,
        legacy=False,
    )

    return pred 