
"""
    Object Detector
        uses TF2 object detection API
        supports multiple backends like Faster R-CNN, SSD and EfficientDet 
        checkpoints written to ~/data/multitracker/checkpoints/$project_name$/bbox
        tensorboard visualization shows loss plots for training and testing and visualizations with drawn ground truth boxes and predictions with confidence scores
"""

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
import random 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py
import tqdm
import json 
import matplotlib.pyplot as plt

from multitracker import util 
from multitracker.keypoint_detection import model 
from multitracker.be import video
from multitracker.object_detection import augmentation

from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

from multitracker.be import dbconnection
db = dbconnection.DatabaseConnection()

# By convention, our non-background classes start counting at 1.  Given
# that we will be predicting just one class, we will therefore assign it a
# `class id` of 1.

# Convert class labels to one-hot; convert everything to tensors.
# The `label_id_offset` here shifts all classes by a certain number of indices;
# we do this here so that the model receives one-hot labels where non-background
# classes start counting at the zeroth index.  This is ordinarily just handled
# automatically in our training binaries, but we need to reproduce it here.
num_classes = 1
label_id_offset = 1
idv_class_id = 1
category_index = {idv_class_id: {'id': 1, 'name': 'animal'}, 2: {'id': 2, 'name': 'gt'}}

def setup_oo_api(models_dir = os.path.expanduser('~/github/models')):
    if not os.path.isdir(models_dir):
        import subprocess
        subprocess.call(['git','clone','--depth','1','https://github.com/tensorflow/models',models_dir])

    print('''    
    please execute
        cd models/research/
        protoc object_detection/protos/*.proto --python_out=.
        cp object_detection/packages/tf2/setup.py .
        python -m pip install .''')
def get_bbox_data(config, vis_input_data=0, video_ids = None):
    if video_ids is None:
        video_ids = config['train_video_ids']

    seed = 2305
    train_image_tensors = []
    train_gt_classes_one_hot_tensors = []
    train_gt_box_tensors = []
    test_image_tensors = []
    test_gt_classes_one_hot_tensors = []
    test_gt_box_tensors = []
    
    frames_dir = os.path.join(video.base_dir_default,'%i' % config['project_id'])
    frame_bboxes = {}
    for _video_id in video_ids.split(','):
        _video_id = int(_video_id)
        db.execute("select * from bboxes where video_id=%i;" % _video_id)
        db_boxxes = [x for x in db.cur.fetchall()]
        random.Random(4).shuffle(db_boxxes)
        for dbbox in db_boxxes:
            _, _, frame_idx, x1, y1, x2, y2 = dbbox
            _key = '%i_%s' % (_video_id, frame_idx) 
            if not _key in frame_bboxes:
                frame_bboxes[_key] = [] 
            frame_bboxes[_key].append(np.array([float(z) for z in [y1,x1,y2,x2]]))
        
        for i, _key in enumerate(frame_bboxes.keys()):
            frame_bboxes[_key] = np.array(frame_bboxes[_key]) 
        
    # read one arbitray frame
    sample_fim = ''
    while not os.path.isfile(sample_fim):
        k = int(np.random.uniform(len(list(frame_bboxes.keys()))))
        sample_fim = os.path.join(frames_dir, video_ids.split(',')[0],'frames','train','%s.png' % list(frame_bboxes.keys())[k].split('_')[1])

    H,W,_ = cv.imread( sample_fim ).shape
    frames = list(frame_bboxes.keys())
    #random.Random(4).shuffle(frames)
    frames = sorted(frames)
    for i, _key in enumerate(frames): # key is video_id_frame_idx 
        frame_bboxes[_key] = frame_bboxes[_key] / np.array([H,W,H,W]) 
        bboxes = frame_bboxes[_key]
        
        if i % 10 > 0: #np.random.uniform() > 0.2:
            train_image_tensors.append(_key) # str(frame_idx).zfill(5)
            train_gt_box_tensors.append(tf.convert_to_tensor(bboxes, dtype=tf.float32))
            train_gt_classes_one_hot_tensors.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[bboxes.shape[0]], dtype=np.int32) - label_id_offset), num_classes))
        else:
            test_image_tensors.append(_key)
            test_gt_box_tensors.append(tf.convert_to_tensor(bboxes, dtype=tf.float32))
            test_gt_classes_one_hot_tensors.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[bboxes.shape[0]], dtype=np.int32) - label_id_offset), num_classes))
    
    # maybe use only a part of the train set
    ddata_train, ddata_test = [], []
    for i in range(len(train_image_tensors)):
        if not ('data_ratio' in config and np.random.uniform() > config['data_ratio']):
            ddata_train.append(train_image_tensors[i])

    for i in range(len(test_image_tensors)):
        ddata_test.append(test_image_tensors[i])
    print('[*] loaded object detection %s data: training on %i samples, testing on %i samples' % (config['object_detection_backbone'], len(ddata_train),len(ddata_test)))
    labeling_list_train = tf.data.Dataset.from_tensor_slices(ddata_train)
    labeling_list_test = tf.data.Dataset.from_tensor_slices(ddata_test)
    
    @tf.function
    def load_im(_key):
        _video_id = tf.strings.split(_key,'_')[0]
        frame_idx = tf.strings.split(_key,'_')[1]
        image_file = tf.strings.join([frames_dir, '/', _video_id,'/frames/train/', frame_idx,'.png'])
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        image = tf.image.resize(image,[H//2,W//2])
        return _key, image

    data_train = labeling_list_train.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).shuffle(256).batch(config['object_detection_batch_size']).prefetch(4*config['object_detection_batch_size'])#.cache()
    data_test = labeling_list_test.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).shuffle(256).batch(config['object_detection_batch_size']).prefetch(4*config['object_detection_batch_size'])#.cache()
    return frame_bboxes, data_train, data_test  
    

def get_pipeline_config(config, pipeline_dir = os.path.join(dbconnection.base_data_dir, 'object_detection')):
    return  os.path.join(pipeline_dir, config['object_detection_backbonepath'], 'pipeline.config')
    
def load_trained_model(config):
    from object_detection.utils import config_util
    from object_detection.builders import model_builder
        
    configs = config_util.get_configs_from_pipeline_file(get_pipeline_config(config))
    model_config = configs['model']
    
    if config['object_detection_backbone'] == 'ssd':
        model_config.ssd.num_classes = 1
    elif config['object_detection_backbone'] == 'fasterrcnn':
        model_config.faster_rcnn.num_classes = 1
    else:
        model_config.ssd.num_classes = 1
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(detection_model=detection_model)
    ckpt.restore(tf.train.latest_checkpoint(config['objectdetection_model'])).expect_partial()
    return detection_model

def restore_weights(config, checkpoint_path = None, gt_boxes = None, gt_classes =None):
    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)
    num_classes = 1
    pipeline_config = get_pipeline_config(config)
    if not os.path.isfile(pipeline_config):
        # download pretrained model 
        url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/%s.tar.gz'%config['object_detection_backbonepath']
        print('[*] downloading pretrained model from',url)
        import subprocess 
        if not os.path.isdir(os.path.join(dbconnection.base_data_dir, 'object_detection')):
            os.makedirs(os.path.join(dbconnection.base_data_dir, 'object_detection'))
        zipped_file = os.path.join(dbconnection.base_data_dir, 'object_detection','%s.tar.gz'%config['object_detection_backbonepath'])
        subprocess.call(['wget','-O', zipped_file,url])
        subprocess.call(['tar','-xyf',zipped_file,'--directory',os.path.join(dbconnection.base_data_dir, 'object_detection')])

    if checkpoint_path is None:
        checkpoint_path = os.path.join(dbconnection.base_data_dir, 'object_detection/%s/checkpoint/ckpt-0.index' % config['object_detection_backbonepath'] )
    
    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just
    # one (for our new rubber ducky class).
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    if config['object_detection_backbone'] == 'ssd':
        model_config.ssd.num_classes = num_classes
        model_config.ssd.freeze_batchnorm = True
    elif config['object_detection_backbone'] == 'fasterrcnn':
        model_config.faster_rcnn.num_classes = num_classes
        #model_config.faster_rcnn.freeze_batchnorm = True
    else:
        model_config.ssd.num_classes = num_classes

    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    if config['object_detection_backbone'] == 'ssd':
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
    
    elif config['object_detection_backbone'] == 'fasterrcnn':
    
        detection_model._extract_proposal_features(detection_model.preprocess(tf.zeros([2, 640, 640, 3]))[0])
        detection_model._extract_box_classifier_features(tf.zeros((2,4,4,1088)))
        fake_model = detection_model.restore_from_objects('detection')['model']
        ckpt = tf.train.Checkpoint(model=fake_model)
        ckpt.restore(checkpoint_path).expect_partial()

    elif config['object_detection_backbone'] == 'efficient':
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
    detection_model.provide_groundtruth(
                    groundtruth_boxes_list=[tf.zeros((7,4))],
                    groundtruth_classes_list=[tf.zeros((7,1))])

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('[*] object detection weights restored from %s' % checkpoint_path)
    return ckpt, model_config, detection_model


def finetune(config, checkpoint_directory, checkpoint_restore = None):
    #setup_oo_api() 
    if not 'finetune' in config:
        config['finetune'] = False 

    if not os.path.isdir(checkpoint_directory): os.makedirs(checkpoint_directory)

    # write config as JSON
    file_json = os.path.join(checkpoint_directory,'config.json')
    with open(file_json, 'w') as f:
        json.dump(config, f, indent=4)

    # load and prepare data 
    frame_bboxes, data_train, data_test = get_bbox_data(config)
    frame_idx, _ = next(iter(data_train))
    gt_boxes, gt_classes = [],[]
    for ii in range(len(frame_idx)):
        gt_boxes.append(tf.convert_to_tensor(frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )], dtype=tf.float32))
        gt_classes.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )].shape[0]], dtype=np.int32) - label_id_offset), num_classes))
    #print('gt_boxes',gt_boxes,'gt_classes',gt_classes)
    ckpt, model_config, detection_model = restore_weights(config, checkpoint_restore, gt_boxes, gt_classes)
    if checkpoint_restore is None:
        tf.keras.backend.set_learning_phase(True)

        # These parameters can be tuned; since our training set has 5 images
        # it doesn't make sense to have a much larger batch size, though we could
        # fit more examples in memory if we wanted to.
        learning_rate = config['lr_objectdetection']
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
            @tf.function(experimental_relax_shapes=True)
            def train_step_fn(image_tensors,
                            groundtruth_boxes_list,
                            groundtruth_classes_list):
                """A single training iteration.

                Args:
                    image_tensors: A list of [B, height, width, 3] Tensor of type tf.float32.
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
                model.provide_groundtruth(
                    groundtruth_boxes_list=groundtruth_boxes_list,
                    groundtruth_classes_list=groundtruth_classes_list)
                image_tensors = tf.concat(image_tensors,axis=0)
                preprocessed_images, shapes = detection_model.preprocess(image_tensors) 

                with tf.GradientTape() as tape:
                    prediction_dict = model.predict(preprocessed_images, shapes)
                    losses_dict = model.loss(prediction_dict, shapes)
                    total_loss = 0.0
                    for v in losses_dict.values():
                        total_loss += v 
                    gradients = tape.gradient(total_loss, vars_to_fine_tune)
                    optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
                return prediction_dict, shapes, total_loss

            @tf.function(experimental_relax_shapes=True)
            def test_step_fn(image_tensors,
                            groundtruth_boxes_list,
                            groundtruth_classes_list):
                """A single training iteration.

                Args:
                    image_tensors: A list of [B, height, width, 3] Tensor of type tf.float32.
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
                model.provide_groundtruth(
                    groundtruth_boxes_list=groundtruth_boxes_list,
                    groundtruth_classes_list=groundtruth_classes_list)
                image_tensors = tf.concat(image_tensors,axis=0)
                preprocessed_images, shapes = detection_model.preprocess(image_tensors) 

                prediction_dict = model.predict(preprocessed_images, shapes)
                losses_dict = model.loss(prediction_dict, shapes)
                total_loss = 0.0
                for v in losses_dict.values():
                    total_loss += v 
                return prediction_dict, shapes, total_loss

            return train_step_fn, test_step_fn

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        train_step_fn, test_step_fn = get_model_train_step_function(detection_model, optimizer, to_fine_tune)
        writer_train = tf.summary.create_file_writer(checkpoint_directory+'/train')
        writer_test = tf.summary.create_file_writer(checkpoint_directory+'/test')
        csv_train = os.path.join(checkpoint_directory,'train_log.csv')
        csv_test = os.path.join(checkpoint_directory,'test_log.csv')

        print('Start fine-tuning!', flush=True)
        early_stopping = False 
        idx = 0
        epoch = 0
        test_losses = []

        def get_vis(vis, _gt_boxes, _prediction_dict):
            
            vis_gt_boxes = np.zeros((len(_gt_boxes), 100, 4))
            for ib in range(len(_gt_boxes)):
                for ij in range(len(_gt_boxes[ib])):
                    vis_gt_boxes[ib,ij,:] =  _gt_boxes[ib][ij]
            #print('vis_gt_scores',vis_gt_scores)

            vis_gt_classes = np.zeros((len(_gt_boxes),100),'uint32')
            for ib in range(len(_gt_boxes)):
                for ij in range(len(_gt_boxes[ib])):
                    vis_gt_classes[ib][ij] = 2
            
            vis_gt_scores = np.zeros((len(_gt_boxes),100))
            for ib in range(len(_gt_boxes)):
                for ij in range(len(_gt_boxes[ib])):
                    vis_gt_scores[ib][ij] = 1.

            vis = viz_utils.draw_bounding_boxes_on_image_tensors(vis, 
                                vis_gt_boxes, # (4, 100, 4) detection_classes (4, 100) (4, 100)
                                vis_gt_classes,
                                vis_gt_scores, #np.ones((4,len(gt_boxes), 100)),
                                category_index)
            vis = tf.cast(vis,tf.uint8)
            vis = viz_utils.draw_bounding_boxes_on_image_tensors(vis, #tf.cast(tf.concat(image_tensors,axis=0),tf.uint8), #vis,
                                _prediction_dict['detection_boxes'],
                                _prediction_dict['detection_classes'].numpy().astype(np.uint32) + label_id_offset,#prediction_dict['detection_classes'],#prediction_dict['detection_classes'].astype(tf.int32) + label_id_offset,
                                _prediction_dict['detection_scores'],
                                category_index)
            vis = tf.cast(vis,tf.float32)
            return vis 

        for epoch in range(int(1e6)):
            for frame_idx, image_tensors in data_train:
                gt_boxes, gt_classes = [],[]
                for ii in range(len(frame_idx)):
                    gt_boxes.append(tf.convert_to_tensor(frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )], dtype=tf.float32))
                    gt_classes.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )].shape[0]], dtype=np.int32) - label_id_offset), num_classes))
                
                # <augmentation>
                image_tensors, gt_boxes, gt_classes = augmentation.augment(config, image_tensors, gt_boxes, gt_classes)
                # </augmentation>

                # Training step (forward pass + backwards pass)
                prediction_dict, shapes, total_loss = train_step_fn(image_tensors, gt_boxes, gt_classes)

                if idx % 100 == 0:
                    # write tensorboard summary
                    with writer_train.as_default():
                        tf.summary.scalar("loss",total_loss,step=idx)
                        
                        prediction_dict = detection_model.postprocess(prediction_dict, shapes)
                        vis = get_vis(tf.cast(tf.concat(image_tensors,axis=0),tf.uint8), gt_boxes, prediction_dict)
                        tf.summary.image('object detection',vis/255.,step=idx)
                        writer_train.flush()
                        with open(csv_train,'a+') as ftrain:
                            ftrain.write('%i,%f\n' % (idx, total_loss))

                ## Test images
                if idx % 250 == 0:
                    num_test_batches = 8
                    test_loss = 0.
                    for frame_idx, image_tensors_test in data_test:
                        gt_boxes_test, gt_classes_test = [],[]
                        for ii in range(len(frame_idx)):
                            gt_boxes_test.append(tf.convert_to_tensor(frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )], dtype=tf.float32))
                            gt_classes_test.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )].shape[0]], dtype=np.int32) - label_id_offset), num_classes))
                        # Test step (forward pass only)
                        prediction_dict_test, shapes, _loss_test = test_step_fn(image_tensors_test, gt_boxes_test, gt_classes_test)
                        test_loss = test_loss + _loss_test/num_test_batches
                    with open(csv_test,'a+') as ftest:
                        ftest.write('%i,%f\n' % (idx, test_loss))

                    test_losses.append(test_loss)

                    # write tensorboard summary
                    with writer_test.as_default():
                        tf.summary.scalar("loss",test_loss,step=idx)

                        prediction_dict_test = detection_model.postprocess(prediction_dict_test, shapes)
                        vis_test = get_vis(tf.cast(tf.concat(image_tensors_test,axis=0),tf.uint8), gt_boxes_test, prediction_dict_test)
                        tf.summary.image('object detection',vis_test/255.,step=idx)
                        writer_test.flush()

                    # check for early stopping -> stop training if test loss is increasing
                    if idx==config['maxsteps_objectdetection'] or (idx>config['minsteps_objectdetection'] and config['early_stopping'] and len(test_losses) > 3 and test_loss > test_losses[-2] and test_loss > test_losses[-3] and test_loss > test_losses[-4] and min(test_losses[:-1]) < 1.5*test_losses[-1]):
                        early_stopping = True 
                        print('[*] stopping object detection early at step %i, epoch %i, because current test loss %f is higher than previous %f and %f' % (idx, epoch, test_loss, test_losses[-2], test_losses[-3]))
                        ckpt_saver = tf.compat.v2.train.Checkpoint(detection_model=detection_model)
                        ckpt_manager = tf.train.CheckpointManager(ckpt_saver, checkpoint_directory, max_to_keep=5)
                        saved_path = ckpt_manager.save()
                        print('[*] saved object detection model to',checkpoint_directory,'->',saved_path)
                        return detection_model
                    if idx % 5000 ==0 :
                        ckpt_saver = tf.compat.v2.train.Checkpoint(detection_model=detection_model)
                        ckpt_manager = tf.train.CheckpointManager(ckpt_saver, checkpoint_directory, max_to_keep=5)
                        saved_path = ckpt_manager.save()
                        print('[*] saved object detection model to',checkpoint_directory,'->',saved_path)
                        
                idx += 1 
        