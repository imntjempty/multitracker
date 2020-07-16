
"""

    python3.7 -m multitracker.object_detection.finetune --project_id 7 --video_id 9 --minutes 0

"""

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py
import tqdm

import matplotlib.pyplot as plt
import random 
from random import shuffle 
from multitracker import util 
from multitracker.keypoint_detection import model 
from multitracker.be import video

from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

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

def get_bbox_data(config, vis_input_data=0):
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
    shuffle(db_boxxes)
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
        gt_box_tensors.append(tf.convert_to_tensor(bboxes, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(np.ones(shape=[bboxes.shape[0]], dtype=np.int32) - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))
        '''for j, bbox in enumerate(bboxes):
            #gt_box_np = np.array(bbox)
            gt_box_np = bbox 

            gt_box_tensors.append(tf.convert_to_tensor(gt_box_np, dtype=tf.float32))
            zero_indexed_groundtruth_classes = tf.convert_to_tensor(np.ones(shape=[gt_box_np.shape[0]], dtype=np.int32) - label_id_offset)
            gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))'''
    print('[*] Done prepping data for %i frames.' % len(frame_bboxes.keys()))
    
    if 0:
        for kk in frame_bboxes.keys():
            print('bboxes:', kk,'->',len(frame_bboxes[kk]),frame_bboxes[kk].shape)

    if vis_input_data:
        plt.figure(figsize=(30, 15))
        for idx in range(9):
            plt.subplot(3, 3, idx+1)
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

def restore_weights():
    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)
    num_classes = 1
    pipeline_config = os.path.expanduser('~/github/models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config')
    checkpoint_path = os.path.expanduser('~/data/multitracker/object_detection/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0.index')

    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression.  We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
    fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')

    return model_config, detection_model

def inference_train_video(detection_model,config, steps, minutes = 0):
    project_id = int(config['project_id'])
    output_dir = '/tmp/multitracker/object_detection/predictions/%i/%i' % (config['video_id'], steps)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('[*] writing %f minutes of video %i frames to %s for step %i' % (minutes,config['video_id'],output_dir,steps))

    frames_dir = os.path.expanduser('~/data/multitracker/projects/%i/%i/frames/train' % (config['project_id'], config['video_id']))
    frame_files = sorted(glob(os.path.join(frames_dir,'*.png')))
    if len(frame_files) == 0:
        raise Exception("ERROR: no frames found in " + str(frames_dir))
    
    if minutes> 0:
        frame_files = frame_files[:int(30. * 60. * minutes)]

    # Again, uncomment this decorator if you want to run inference eagerly
    @tf.function
    def detect(input_tensor):
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
        preprocessed_image, shapes = detection_model.preprocess(input_tensor)
        prediction_dict = detection_model.predict(preprocessed_image, shapes)
        return detection_model.postprocess(prediction_dict, shapes)

    label_id_offset = 1
    idv_class_id = 1
    num_classes = 1

    category_index = {idv_class_id: {'id': idv_class_id, 'name': 'animal'}}
    frame_detections = {}
    for i in tqdm.tqdm(range(len(frame_files))):
        #images = []
        image = cv.imread(frame_files[i])
        input_tensor = tf.convert_to_tensor(np.expand_dims(image,axis=0), dtype=tf.float32)
        detections = detect(input_tensor)
        frame_detections[frame_files[i]] = detections
        #print(i,'/',len(frame_files),frame_files[i])
        plot_detections(
            image,
            detections['detection_boxes'][0].numpy(),
            detections['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset,
            detections['detection_scores'][0].numpy(),
            category_index, figsize=(15, 20), image_name=os.path.join(output_dir,"frame_" + ('%06d' % i) + ".png"))

    file_bboxes = output_dir + '_bboxes'
    np.savez_compressed(file_bboxes,boxes=frame_detections)
    print('[*] saved',file_bboxes)

def main(args):
    #setup_oo_api() 

    config = model.get_config(project_id = args.project_id)
    config['project_id'] = args.project_id
    config['video_id'] = args.video_id
    config['finetune'] = args.finetune

    # load and prepare data 
    train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors = get_bbox_data(config)
    model_config, detection_model = restore_weights()

    tf.keras.backend.set_learning_phase(True)

    # These parameters can be tuned; since our training set has 5 images
    # it doesn't make sense to have a much larger batch size, though we could
    # fit more examples in memory if we wanted to.
    batch_size = 8
    learning_rate = 0.001
    num_batches = int(1e7) # was 100 with 5 examples

    # Select variables in top layers to fine-tune.
    trainable_variables = detection_model.trainable_variables
    
    to_fine_tune = []
    if config['finetune']:
        prefixes_to_train = [
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
            'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
        for var in trainable_variables:
            if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
                to_fine_tune.append(var)
    else:
        to_fine_tune = trainable_variables

    # Set up forward + backward pass for a single train step.
    def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
        """Get a tf.function for training step."""

        # Use tf.function for a bit of speed.
        # Comment out the tf.function decorator if you want the inside of the
        # function to run eagerly.
        @tf.function
        def train_step_fn(image_tensors,
                        groundtruth_boxes_list,
                        groundtruth_classes_list):
            """A single training iteration.

            Args:
                image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
                groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
                groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

            Returns:
                A scalar tensor representing the total loss for the input batch.
            """
            shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
            model.provide_groundtruth(
                groundtruth_boxes_list=groundtruth_boxes_list,
                groundtruth_classes_list=groundtruth_classes_list)
            with tf.GradientTape() as tape:
                preprocessed_images = tf.concat(
                    [detection_model.preprocess(image_tensor)[0]
                    for image_tensor in image_tensors], axis=0)
                prediction_dict = model.predict(preprocessed_images, shapes)
                losses_dict = model.loss(prediction_dict, shapes)
                total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
                gradients = tape.gradient(total_loss, vars_to_fine_tune)
                optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
            return total_loss

        return train_step_fn

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(detection_model, optimizer, to_fine_tune)

    print('Start fine-tuning!', flush=True)
    for idx in range(num_batches):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(train_image_tensors)))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]

        # Note that we do not do data augmentation in this demo.  If you want a
        # a fun exercise, we recommend experimenting with random horizontal flipping
        # and random cropping :)
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        #print('gt_boxes_list',gt_boxes_list)
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        image_tensors = [train_image_tensors[key] for key in example_keys]

        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(num_batches) + ', loss=' +  str(total_loss.numpy()), flush=True)

        '''if idx % 1000 == 0:
            finetuned_checkpoint_path = os.path.expanduser('~/checkpoints/object_detection')
            finetuned_checkpoint_path = os.path.join(finetuned_checkpoint_path,'finetuned_%i.h5' % idx)
            detection_model.save(finetuned_checkpoint_path)
            print('[*] saved model to', finetuned_checkpoint_path)'''

        if idx > 0 and (idx % 15000 == 0):# or idx in [5000]):
            inference_train_video(detection_model,config,idx,args.minutes)
    print('Done fine-tuning!')

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    parser.add_argument('--minutes',required=False,default=0.0,type=float)
    #parser.add_argument('--thresh_detection',required=False,default=0.5,type=float)
    parser.add_argument('--finetune', dest='finetune', default=False, action='store_true')
    args = parser.parse_args()
    main(args)