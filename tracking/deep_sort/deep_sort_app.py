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
from multitracker.keypoint_detection import roi_segm
from multitracker.tracking import inference
from multitracker.tracking.keypoint_tracking import tracker as keypoint_tracking
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


global count_failed
global count_ok
count_ok, count_failed =0,0

def append_crop_mosaic(frame, vis_crops):
    # stack output image together
    # left side: original visualization with boxes and tracks
    # right side: moasics focused on tracks with keypoints
    if len(vis_crops) == 0:
        return np.hstack((frame,np.zeros((frame.shape[0],frame.shape[0],3),'uint8')))

    n = int(np.sqrt(len(vis_crops)))
    if not len(vis_crops) == int(n+0.5) ** 2: # is not square root?
        n += 1 
    n = max(2,n)

    d = int(frame.shape[0]/n)
    mosaic = np.zeros((frame.shape[0],frame.shape[0],3),'uint8')
    for i in range(len(vis_crops)):
        vis_crops[i] = cv.resize(vis_crops[i],(d,d))
        cx = i%n
        cy = i//n 
        mosaic[d*cy:d*cy+d,d*cx:d*cx+d,:] = vis_crops[i]

    out = np.hstack((frame, mosaic))
    return out

#def run(config, detection_file, output_file, min_confidence,
#        nms_max_overlap, max_cosine_distance,
#        nn_budget, display):
def run(config, detection_model, encoder_model, keypoint_model, output_dir, min_confidence, min_confidence_keypoints, crop_dim, 
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
    
    max_age = 30 
    #max_age = 5
    config['count'] = 0 
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    if 'tracking_method' in config or config['tracking_method'] == 'DeepSORT':
        if 'fixed_number' in config and config['fixed_number'] is not None:
            tracker = Tracker(metric,fixed_number=config['fixed_number'])
        else:
            tracker = Tracker(metric)
    elif config['tracking_method'] == 'Tracktor++':
        #if 'fixed_number' in config and config['fixed_number'] is not None:
        tracker = tracktor.Tracker()
    else:
        raise Exception("Please give a method for tracking (supported DeepSORT and Tracktor++)")

    results = []

    if not os.path.isdir('/tmp/vis/'): os.makedirs('/tmp/vis/')

    if 'video' in config and config['video'] is not None:
        video_reader = cv.VideoCapture( config['video'] )
    else:
        video_reader = None 
    
    [Hframe,Wframe,_] = cv.imread(glob(os.path.join(os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'test'),'*.png'))[0]).shape
    
    video_file_out = inference.get_video_output_filepath(config)
    if os.path.isfile(video_file_out): os.remove(video_file_out)
    import skvideo.io
    video_writer = skvideo.io.FFmpegWriter(video_file_out, outputdict={
        '-vcodec': 'libx264',  #use the h.264 codec
        '-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
    }) 
    print('[*] writing video file %s' % video_file_out)
    
    keypoint_tracker = keypoint_tracking.KeypointTracker()
    

    def frame_callback(vis, frame_idx):
        global count_failed
        global count_ok
        #print("Processing frame %05d" % frame_idx)
        frame_directory = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'train')
        frame_file = os.path.join(frame_directory, '%05d.png' % frame_idx)
        
        if video_reader is not None and video_reader.isOpened():
            ret, frame = video_reader.read()
        else:
            frame = cv.imread(frame_file)
        frame_, detections = inference.detect_frame_boundingboxes(config, detection_model, encoder_model, seq_info, frame, frame_idx)
            
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
        
        keypoints = inference.inference_keypoints(config, frame, detections, keypoint_model, crop_dim, min_confidence_keypoints)
        # update tracked keypoints with new detections
        tracked_keypoints = keypoint_tracker.update(keypoints)

        # Update visualization.
        if len(seq_info["image_filenames"][1])>int(frame_idx):
            #image = cv.imread(seq_info["image_filenames"][1][int(frame_idx)], cv.IMREAD_COLOR)
            image=frame
            # draw keypoint detections 'whitish'
            im = np.array(frame, copy=True)
            
            radius_keypoint = 5
            # draw history of keypoint track 
            for i in range( len(tracked_keypoints) ):
                color_keypoint = [int(ss) for ss in colors[tracked_keypoints[i].history_class[-1]%len(colors)]]
                for j in range( 1, len(tracked_keypoints[i].history_estimated)):
                    p1 = tuple(np.int32(np.around(tracked_keypoints[i].history_estimated[j])))
                    p2 = tuple(np.int32(np.around(tracked_keypoints[i].history_estimated[j-1])))
                    im = cv.line(im, p1, p2, color_keypoint, 2)

            # draw keypoint tracks in full color
            for keypoint_track in keypoint_tracker.get_deadnalive_tracks():
                should_draw = True 
                x, y = [ int(round(c)) for c in keypoint_track.position]
                if keypoint_track.alive:
                    color_keypoint = [int(ss) for ss in colors[keypoint_track.history_class[-1]%len(colors)]]
                else:
                    color_keypoint = [int(ss)//2+128 for ss in colors[keypoint_track.history_class[-1]%len(colors)]]
                    if len(keypoint_track.history_class) < 10:
                        should_draw = False
                if should_draw:
                    im = cv.circle(im, (x,y), radius_keypoint, color_keypoint, -1)

            # draw detected keypoint    
            for [x,y,c] in keypoints:
                x, y = [int(round(x)),int(round(y))]
                color_keypoint = [int(ss) for ss in colors[c%len(colors)]]
                color_keypoint = [c//2 + 64 for c in color_keypoint]
                im = cv.circle(im, (x,y), radius_keypoint, color_keypoint, 3)
            
            # crop keypointed vis 
            vis_crops = [  ]
            for i, track in enumerate(tracker.tracks):
                if track.is_confirmed():
                    x1,y1,x2,y2 = track.to_tlbr()
                    center = roi_segm.get_center(x1,y1,x2,y2, image.shape[0],image.shape[1], crop_dim)
                    vis_crops.append( im[center[0]-crop_dim//2:center[0]+crop_dim//2,center[1]-crop_dim//2:center[1]+crop_dim//2,:] )    
                #vis_crops[-1] = cv.resize(vis_crops[-1], (im.shape[0]//2,im.shape[0]//2))
            
            _shape = [im.shape[0]//2,im.shape[1]//2]
            if len(vis_crops)>0:
                _shape = vis_crops[0].shape[:2]
            
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)
            
            out = append_crop_mosaic(vis.viewer.image,vis_crops)
        
            try:
                video_writer.writeFrame(cv.cvtColor(out, cv.COLOR_BGR2RGB)) #out[:,:,::-1])
                count_ok += 1
                config['count'] += 1
            except Exception as e:
                count_failed += 1 
                print('[*] Video Writing for frame_idx %s failed. image shape'%frame_idx,out.shape)
                print(e)

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

    if 0 and display:
        print('[*] finished frames, writing video ...')
        file_video = '/tmp/tracking.mp4'
        subprocess.call(['ffmpeg','-framerate','30','-i','/tmp/vis/%d.png', '-vf', 'format=yuv420p','-vcodec','libx265', file_video])
        subprocess.call(['rm','/tmp/vis/*.png'])
        print('[*] wrote video to %s' % file_video)
    #video_writer.release() 

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
