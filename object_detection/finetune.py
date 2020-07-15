
import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py

import matplotlib.pyplot as plt

from multitracker import util 
from multitracker.keypoint_detection import model 
from multitracker.be import video

from object_detection.utils import visualization_utils as viz_utils

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

def setup_oo_api(models_dir = os.path.expanduser('~/github/models')):
    if not os.path.isdir(models_dir):
        subprocess.call(['git','clone','--depth','1','https://github.com/tensorflow/models',models_dir])

    print('''    
    please execute
        cd models/research/
        protoc object_detection/protos/*.proto --python_out=.
        cp object_detection/packages/tf2/setup.py .
        python -m pip install .''')

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)

def get_bbox_data(config, vis_input_data=True):
    # By convention, our non-background classes start counting at 1.  Given
    # that we will be predicting just one class, we will therefore assign it a
    # `class id` of 1.
    idv_class_id = 1
    num_classes = 1

    category_index = {idv_class_id: {'id': idv_class_id, 'name': 'animal'}}

    # Convert class labels to one-hot; convert everything to tensors.
    # The `label_id_offset` here shifts all classes by a certain number of indices;
    # we do this here so that the model receives one-hot labels where non-background
    # classes start counting at the zeroth index.  This is ordinarily just handled
    # automatically in our training binaries, but we need to reproduce it here.
    label_id_offset = 1
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []



    frames_dir = os.path.join(video.get_frames_dir(video.get_project_dir(video.base_dir_default, config['project_id']), config['video_id']),'train')
    frame_bboxes = {}
    db.execute("select * from bboxes where video_id=%i;" % config['video_id'])
    db_boxxes = [x for x in db.cur.fetchall()]
    for dbbox in db_boxxes:
        _, _, frame_idx, x1, y1, x2, y2 = dbbox 
        if not frame_idx in frame_bboxes:
            frame_bboxes[frame_idx] = [] 
        frame_bboxes[frame_idx].append(np.array([float(z) for z in [y1,x1,y2,x2]]))
    
    for i, frame_idx in enumerate(frame_bboxes.keys()):
        frame_bboxes[frame_idx] = np.array(frame_bboxes[frame_idx]) 
    
    for i, frame_idx in enumerate(frame_bboxes.keys()):
        f = os.path.join(frames_dir, '%s.png' % frame_idx)
        train_image_np = cv.imread(f)
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(train_image_np, dtype=tf.float32), axis=0))
        
        H, W = train_image_np.shape[:2]
        frame_bboxes[frame_idx] = frame_bboxes[frame_idx] / np.array([H,W,H,W]) 
        bboxes = frame_bboxes[frame_idx]
        for j, bbox in enumerate(bboxes):
            #gt_box_np = np.array(bbox)
            gt_box_np = bbox 

            gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
            zero_indexed_groundtruth_classes = tf.convert_to_tensor(np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
            gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))
    print('[*] Done prepping data for %i frames.' % len(frame_bboxes.keys()))

    if vis_input_data:
        plt.figure(figsize=(30, 15))
        for idx in range(10):
            plt.subplot(2, 3, idx+1)
            f = os.path.join(frames_dir, '%s.png' % sorted(list(frame_bboxes.keys()))[idx])
            fo = '/tmp/oo_%s'%f.split('/')[-1]
            train_image_np = cv.imread(f)
            
            gt_boxes = frame_bboxes[sorted(list(frame_bboxes.keys()))[idx]]
            classes = np.ones(shape=[gt_boxes.shape[0]], dtype=np.int32)
            dummy_scores = np.ones(shape=[classes.shape[0]], dtype=np.float32)  # give boxes a score of 100%
            plot_detections(
                train_image_np,
                gt_boxes,
                classes.reshape(gt_boxes.shape[0]),
                dummy_scores.reshape(gt_boxes.shape[0]), 
                category_index, image_name = fo)
            print('[*] wrote input data vis %s' % fo)
    return train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors

def main(args):
    #setup_oo_api() 

    config = model.get_config(project_id = args.project_id)
    config['project_id'] = args.project_id
    config['video_id'] = args.video_id

    # load and prepare data 
    train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors = get_bbox_data(config)


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    parser.add_argument('--minutes',required=False,default=0.0,type=float)
    #parser.add_argument('--thresh_detection',required=False,default=0.5,type=float)
    args = parser.parse_args()
    main(args)