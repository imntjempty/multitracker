# vim: expandtab:ts=4:sw=4
# python3.7 -m multitracker.tracking.deep_sort.deep_sort_app --detection_file '/tmp/multitracker/object_detection/predictions/9/60000_bboxes_*.npz' --project_id 7 --video_id 9

from __future__ import division, print_function, absolute_import

import argparse
import os
from glob import glob 
import subprocess

import cv2 as cv 
import numpy as np
import tensorflow as tf 
assert tf.__version__.startswith('2.'), 'YOU MUST INSTALL TENSORFLOW 2.X'
print('[*] TF version',tf.__version__)
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import Model

from multitracker.tracking.deep_sort.application_util import preprocessing
from multitracker.tracking.deep_sort.application_util import visualization
from multitracker.tracking.deep_sort.deep_sort import nn_matching
from multitracker.tracking.deep_sort.deep_sort.detection import Detection
from multitracker.tracking.deep_sort.deep_sort.tracker import Tracker
from multitracker import autoencoder
from multitracker.keypoint_detection import roi_segm, unet
from multitracker.tracking.inference import get_heatmaps_keypoints
from multitracker.be import video
from multitracker import util 

colors = util.get_colors()

def gather_sequence_info(config):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'train')
    image_filenames = {
        1: sorted(glob(os.path.join(image_dir,'*.png')))
    }

    groundtruth = None
    
    min_frame_idx = image_filenames[1][0].split('/')[-1].split('.')[0]
    max_frame_idx = image_filenames[1][-1].split('/')[-1].split('.')[0]
    print('[*] min_frame_idx',min_frame_idx,'max_frame_idx',max_frame_idx)

    if len(image_filenames) > 0:
        for iv,v in enumerate(image_filenames.values()):
            if iv < 5:
                print(v[:5])
        image_size = cv.imread(image_filenames[list(image_filenames.keys())[0]][0]  ,cv.IMREAD_GRAYSCALE).shape
         
    else:
        image_size = None

    feature_dim = 128#detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": config['project_name'],
        "image_filenames": image_filenames,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def load_feature_extractor(config):
    ## feature extractor
    config.update({'img_height':640, 'img_width': 640})
    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 3])
    feature_extractor,encoder = autoencoder.Encoder(inputs)
    encoder_model = Model(inputs = inputs, outputs = [feature_extractor,encoder])
    ckpt = tf.train.Checkpoint(encoder_model=encoder_model)

    ckpt_manager = tf.train.CheckpointManager(ckpt, config['autoencoder_model'], max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)
    else:
        print('[*] WARNING: could not load pretrained model!')
    return encoder_model

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
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    #print('preprocessed_image',preprocessed_image.shape,'shapes',shapes)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    detections['detection_boxes'] = detections['detection_boxes'][0,:,:]
    detections['detection_scores'] = detections['detection_scores'][0]
    return detections

act_detection_part_file = None
act_detection_part_data = None 
def load_detections(config, detection_model, encoder_model, seq_info, frame_idx, thresh_detection = 0.8):
    frame_directory = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'train')
    frame_file = os.path.join(frame_directory, '%05d.png' % frame_idx)
    
    inp_tensor = cv.imread(frame_file)
    inp_tensor = tf.expand_dims(inp_tensor,0)
    features, _ = encoder_model(autoencoder.preprocess(inp_tensor),training=False)
    
    bboxes = detect_bounding_boxes(detection_model, inp_tensor)
    #for i in range(bboxes['detection_boxes'].shape[0]):
    results = []
    for j in range(bboxes['detection_boxes'].shape[0]):
        #print(i,j,'bbox',frame_idx,bboxes['detection_boxes'][i].shape,bboxes['detection_scores'][i].shape)
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

            _scale = inp_tensor.shape[2]/1920
            features_crop = features[:,int(_scale*top/8.):int(_scale*(top+height)/8.),int(_scale*left/8.):int(_scale*(left+width)/8.),:] 
            #print('[*] cropped',int(_scale*top/8.),int(_scale*(top+height)/8.),int(_scale*left/8.),int(_scale*(left+width)/8.),'->',features_crop.shape)
            features_crop = tf.keras.layers.GlobalAveragePooling2D()(features_crop)
            features_crop = features_crop.numpy()[0,:]

            detection = Detection([left,top,width,height], proba, features_crop)
            
            results.append(detection)        

    return inp_tensor[0,:,:,:], results        

#def run(config, detection_file, output_file, min_confidence,
#        nms_max_overlap, max_cosine_distance,
#        nn_budget, display):
def run(config, detection_model, encoder_model, keypoint_model, output_dir, min_confidence,
        nms_max_overlap, max_cosine_distance,
        nn_budget, display):
            
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """

    seq_info = gather_sequence_info(config)
    #print('seq_info',seq_info.keys())
    
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    if 'fixed_number' in config and config['fixed_number'] is not None:
        tracker = Tracker(metric,fixed_number=config['fixed_number'])
    else:
        tracker = Tracker(metric)
    results = []

    if not os.path.isdir('/tmp/vis/'): os.makedirs('/tmp/vis/')

    def frame_callback(vis, frame_idx):
        #print("Processing frame %05d" % frame_idx)
        frame, detections = load_detections(config, detection_model, encoder_model, seq_info, frame_idx)
        #print('frame',frame.dtype,frame.shape)
        #im = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_kp = unet.preprocess(frame)
        crop_dim = get_roi_crop_dim(config['project_id'], config['video_id'], frame.shape[0]) 
        
        #if detections is None:
        #    return False 

        detections = [d for d in detections if d.confidence >= min_confidence]
        #print('[*] processing frame',frame_idx,'with %i detections ' % len(detections))
        #if len(detections) > 10:
        #    detections = []

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        #print('[*] found %i detections' % len(detections))
        # Update tracker.
        tracker.predict()
        tracker.update(detections)
        
        # inference keypoints for all detections
        y_kpheatmaps = np.zeros((frame.shape[0],frame.shape[1],1+len(config['keypoint_names'])),np.float32)
        #for i, track in enumerate(tracker.tracks):
        for i, detection in enumerate(detections):
            #x1,y1,x2,y2 = track.last_detection.to_tlbr()
            x1,y1,x2,y2 = detection.to_tlbr()
            #print('detect',i,x1,y1,x2,y2)
            #x1,y1,x2,y2 = track.to_tlbr()
            # crop region around center of bounding box
            center = roi_segm.get_center(x1,y1,x2,y2, frame.shape[0], frame.shape[1], crop_dim)
            center[0] = int(round(center[0]))
            center[1] = int(round(center[1]))
            roi = frame_kp[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:]
            #roi = tf.image.resize(roi,[config['img_height'],config['img_width']])
            roi = tf.image.resize(roi,[224,224])
            roi = tf.expand_dims(tf.convert_to_tensor(roi),axis=0)
            #yroi = keypoint_model.predict(roi)
            yroi = keypoint_model(roi, training=False).numpy()
            yroi = yroi[0,:,:,:]
            #print('yroi', crop_dim, yroi.dtype, yroi.shape,center[0]-crop_dim//2,center[0]+crop_dim//2,center[1]-crop_dim//2,center[1]+crop_dim//2)
            yroi = cv.resize(yroi,(crop_dim//2*2,crop_dim//2*2))
            
            y_kpheatmaps[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:] = yroi
        keypoints = get_heatmaps_keypoints(y_kpheatmaps, thresh_detection=0.5)
        print('%i keypoints' % len(keypoints),[kp for kp in keypoints])

        # Update visualization.
        if display:
            #print(seq_info["image_filenames"].keys())
            #print('A',seq_info["image_filenames"].keys())
            #print('sev',len(seq_info["image_filenames"][1][int(frame_idx)]))
            image = cv.imread(seq_info["image_filenames"][1][int(frame_idx)], cv.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            
            # draw keypoints 
            im = np.array(vis.viewer.image, copy=True)
            
            color_offset = 10
            radius_keypoint = 10
            for [x,y,c] in keypoints:
                x, y = [int(round(x)),int(round(y))]
                color_keypoint = [int(ss) for ss in colors[c]]
                im = cv.circle(im, (x,y), radius_keypoint, color_keypoint, -1)
            
            # crop keypointed vis 
            vis_crops = [  ]
            for i, track in enumerate(tracker.tracks):
                x1,y1,x2,y2 = track.to_tlbr()
                #x1,y1,x2,y2 = track.last_detection.to_tlbr()
                center = roi_segm.get_center(x1,y1,x2,y2, image.shape[0],image.shape[1], crop_dim)
                vis_crops.append( im[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:] )    
                #if vis_crops[-1] is not None:
                vis_crops[-1] = cv.resize(vis_crops[-1], (im.shape[0]//2,im.shape[0]//2))
            
            #print('len tracks',len(tracker.tracks),'len vis crops',len(vis_crops))
            for i in range(4-len(vis_crops)):
                vis_crops.append( np.zeros((im.shape[0]//2,im.shape[1]//2,3),'uint8'))

            # stack output image together
            # left side: original visualization with boxes and tracks
            # right side: moasics focused on tracks with keypoints
            out = vis.viewer.image
            out = np.hstack((
                out,
                np.vstack((
                    vis_crops[0],
                    vis_crops[2]
                )),
                np.vstack((
                    vis_crops[1],
                    vis_crops[3]
                ))
            ))
            vis.set_image(out.copy())

            if 0 :
                # write drawn keypoint image
                vis_keypoints = np.zeros((y_kpheatmaps.shape[0],y_kpheatmaps.shape[1],3),'uint8')
                for ii in range(len(config['keypoint_names'])):
                    channel = y_kpheatmaps[:,:,ii]
                    print('minmax',channel.min(),channel.max())
                    channel = np.int32(255.*(channel-channel.min())/(channel.max() - channel.min()))
                    vis_keypoints[:,:,ii%3] = channel 
                out = np.hstack((vis_keypoints,out))
            # write image to disk
            fo = '/tmp/vis/%s.png'  %frame_idx
            cv.imwrite(fo,out)
            print('[*] wrote %s' % fo )

            
        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        return True
    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    #f = open(output_file, 'w')
    #for row in results:
    #    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
    #        row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    result_data = {}
    for [frame_idx, track_id,a,b,c,d] in results:
        if not frame_idx in result_data:
            result_data[frame_idx] = []
        result_data[frame_idx].append([track_id,a,b,c,d])
    np.savez_compressed('.'.join(output_file.split('.')[:-1]), tracked_boxes=result_data)
        

    if display:
        print('[*] finished frames, writing video ...')
        file_video = '/tmp/tracking.mp4'
        subprocess.call(['ffmpeg','-framerate','30','-i','/tmp/vis/%d.png', '-vf', 'format=yuv420p','-vcodec','libx265', file_video])
        subprocess.call(['rm','/tmp/vis/*.png'])
        print('[*] wrote video to %s' % file_video)

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--project_id", help="Project ID", default=None,
        required=True)
    parser.add_argument(
        "--video_id", help="Video ID", default=None,
        required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()



if __name__ == "__main__":
    from multitracker.keypoint_detection import model
    config = model.get_config()
    args = parse_args()
    config['project_id'] = args.project_id
    config['video_id'] = args.video_id
    run(
        config, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap,
        args.max_cosine_distance, args.nn_budget, args.display)
