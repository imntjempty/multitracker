
"""

    python3.7 -m multitracker.object_detection.finetune --project_id 7 --video_id 9 --minutes 10

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
category_index = {idv_class_id: {'id': 1, 'name': 'animal'}}

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
  #if image_name:
  #  plt.imsave(image_name, image_np_with_annotations)
  #else:
  #  plt.imshow(image_np_with_annotations)
  return image_np_with_annotations

def get_bbox_data(config, vis_input_data=0):
    train_image_tensors = []
    train_gt_classes_one_hot_tensors = []
    train_gt_box_tensors = []
    test_image_tensors = []
    test_gt_classes_one_hot_tensors = []
    test_gt_box_tensors = []
    
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
    
    H,W,_ = cv.imread( os.path.join(frames_dir, '%s.png' % list(frame_bboxes.keys())[0]) ).shape
    for i, frame_idx in enumerate(frame_bboxes.keys()):
        #print(i,frame_idx)
        f = os.path.join(frames_dir, '%s.png' % frame_idx)

        frame_bboxes[frame_idx] = frame_bboxes[frame_idx] / np.array([H,W,H,W]) 
        bboxes = frame_bboxes[frame_idx]
        
        if np.random.uniform() > 0.2:
            #train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(image_np, dtype=tf.float32), axis=0))
            train_image_tensors.append(frame_idx)
            train_gt_box_tensors.append(tf.convert_to_tensor(bboxes, dtype=tf.float32))
            train_gt_classes_one_hot_tensors.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[bboxes.shape[0]], dtype=np.int32) - label_id_offset), num_classes))
        else:
            #test_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(image_np, dtype=tf.float32), axis=0))
            test_image_tensors.append(frame_idx)
            test_gt_box_tensors.append(tf.convert_to_tensor(bboxes, dtype=tf.float32))
            test_gt_classes_one_hot_tensors.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[bboxes.shape[0]], dtype=np.int32) - label_id_offset), num_classes))
    
    ddata_train, ddata_test = [], []
    for i in range(len(train_image_tensors)):
        ddata_train.append([train_image_tensors[i], train_gt_box_tensors[i], train_gt_classes_one_hot_tensors[i]])
    for i in range(len(test_image_tensors)):
        ddata_test.append([test_image_tensors[i], test_gt_box_tensors[i], test_gt_classes_one_hot_tensors[i]])
    labeling_list_train = tf.data.Dataset.from_tensor_slices(train_image_tensors)
    labeling_list_test = tf.data.Dataset.from_tensor_slices(test_image_tensors)
    
    @tf.function
    def load_im(frame_idx):
        image_file = tf.strings.join([frames_dir, '/',tf.cast(frame_idx,tf.string),'.png'])
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        return frame_idx, image

    data_train = labeling_list_train.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['object_detection_batch_size']).prefetch(4*config['object_detection_batch_size'])#.cache()
    data_test = labeling_list_test.map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['object_detection_batch_size']).prefetch(4*config['object_detection_batch_size'])#.cache()
    data_train = data_train.shuffle(512)
    return frame_bboxes, data_train, data_test  
    

def get_bbox_data_ram(config, vis_input_data=0):
    train_image_tensors = []
    train_gt_classes_one_hot_tensors = []
    train_gt_box_tensors = []
    test_image_tensors = []
    test_gt_classes_one_hot_tensors = []
    test_gt_box_tensors = []

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
        image_np = cv.imread(f)
        image_np = cv.cvtColor(image_np,cv.COLOR_BGR2RGB)
        #image_np = cv.resize(image_np,(640,640))
        
        H, W = image_np.shape[:2]
        frame_bboxes[frame_idx] = frame_bboxes[frame_idx] / np.array([H,W,H,W]) 
        bboxes = frame_bboxes[frame_idx]
        
        if np.random.uniform() > 0.2:
            train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(image_np, dtype=tf.float32), axis=0))
            train_gt_box_tensors.append(tf.convert_to_tensor(bboxes, dtype=tf.float32))
            train_gt_classes_one_hot_tensors.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[bboxes.shape[0]], dtype=np.int32) - label_id_offset), num_classes))
        else:
            test_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(image_np, dtype=tf.float32), axis=0))
            test_gt_box_tensors.append(tf.convert_to_tensor(bboxes, dtype=tf.float32))
            test_gt_classes_one_hot_tensors.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[bboxes.shape[0]], dtype=np.int32) - label_id_offset), num_classes))
        
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

    
    return train_image_tensors, train_gt_box_tensors, train_gt_classes_one_hot_tensors, test_image_tensors, test_gt_box_tensors, test_gt_classes_one_hot_tensors

#def get_pipeline_config(config, pipeline_dir = os.path.expanduser('~/github/models/research/object_detection/configs/tf2')):
#    return  os.path.join(pipeline_dir, config['object_detection_backbonepath'])
def get_pipeline_config(config, pipeline_dir = os.path.join(dbconnection.base_data_dir, 'object_detection')):
    return  os.path.join(pipeline_dir, config['object_detection_backbonepath'], 'pipeline.config')
    
    
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

    detection_model.provide_groundtruth(
                    groundtruth_boxes_list=[tf.zeros((7,4))],
                    groundtruth_classes_list=[tf.zeros((7,1))])

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('[*] object detection weights restored from %s' % checkpoint_path)
    return ckpt, model_config, detection_model

def inference_train_video(detection_model,config, steps, minutes = 0):
    project_id = int(config['project_id'])
    output_dir = '/tmp/multitracker/object_detection/predictions/%i/%i' % (config['video_id'], steps)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    print('[*] writing object detection bounding boxes %f minutes of video %i frames to %s for step %i' % (minutes,config['video_id'],output_dir,steps))

    frames_dir = os.path.join(dbconnection.base_data_dir, 'projects/%i/%i/frames/train' % (config['project_id'], config['video_id']))
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
        #input_tensor = tf.image.resize_with_pad(input_tensor,640,640,antialias=True)
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
        if i>0 and i % 1000 == 0:
            file_bboxes = output_dir + '_bboxes_%05d' % i 
            np.savez_compressed(file_bboxes,boxes=frame_detections)
            #np.save(file_bboxes,frame_detections)
            #print('[*] saved',file_bboxes)
            frame_detections = {}

        try:
            image = cv.imread(frame_files[i])
            input_tensor = tf.convert_to_tensor(np.expand_dims(image,axis=0), dtype=tf.float32)
            detections = detect(input_tensor)
            frame_detections[frame_files[i]] = detections
            #print(i,'/',len(frame_files),frame_files[i])
            if 1:
                plot_detections(
                    image,
                    detections['detection_boxes'][0].numpy(),
                    detections['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset,
                    detections['detection_scores'][0].numpy(),
                    category_index, figsize=(15, 20), image_name=os.path.join(output_dir,"frame_" + ('%06d' % i) + ".png"))

        except Exception as e:
            print(e)

    file_bboxes = output_dir + '_bboxes_%i' % len(frame_files)
    #np.save(file_bboxes,frame_detections)
    np.savez_compressed(file_bboxes,boxes=frame_detections)
    print('[*] saved',file_bboxes)

def finetune(config, checkpoint_directory, checkpoint_restore = None):
    #setup_oo_api() 
    if not 'finetune' in config:
        config['finetune'] = False 

    # load and prepare data 
    #train_image_tensors, train_gt_box_tensors, train_gt_classes_one_hot_tensors, test_image_tensors, test_gt_box_tensors, test_gt_classes_one_hot_tensors = get_bbox_data(config)
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
                            groundtruth_classes_list,
                            update_weights = True):
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
                model.provide_groundtruth(
                    groundtruth_boxes_list=groundtruth_boxes_list,
                    groundtruth_classes_list=groundtruth_classes_list)
                image_tensors = tf.concat(image_tensors,axis=0)
                preprocessed_images, shapes = detection_model.preprocess(image_tensors) 

                if update_weights:
                    with tf.GradientTape() as tape:
                        prediction_dict = model.predict(preprocessed_images, shapes)
                        losses_dict = model.loss(prediction_dict, shapes)
                        total_loss = 0.0
                        for v in losses_dict.values():
                            total_loss += v 
                        gradients = tape.gradient(total_loss, vars_to_fine_tune)
                        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
                else:
                    prediction_dict = model.predict(preprocessed_images, shapes)
                    losses_dict = model.loss(prediction_dict, shapes)
                    total_loss = 0.0
                    for v in losses_dict.values():
                        total_loss += v 
                return prediction_dict, shapes, total_loss

            return train_step_fn

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        train_step_fn = get_model_train_step_function(detection_model, optimizer, to_fine_tune)
        writer_train = tf.summary.create_file_writer(checkpoint_directory+'/train')
        writer_test = tf.summary.create_file_writer(checkpoint_directory+'/test')
    
        print('Start fine-tuning!', flush=True)
        early_stopping = False 
        idx = 0
        epoch = 0
        test_losses = []
        for epoch in range(int(1e6)):
            for frame_idx, image_tensors in data_train:
                gt_boxes, gt_classes = [],[]
                for ii in range(len(frame_idx)):
                    gt_boxes.append(tf.convert_to_tensor(frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )], dtype=tf.float32))
                    gt_classes.append(tf.one_hot(tf.convert_to_tensor(np.ones(shape=[frame_bboxes[str(frame_idx[ii].numpy().decode("utf-8") )].shape[0]], dtype=np.int32) - label_id_offset), num_classes))
                
                # <augmentation>
                if 0:
                    if np.random.uniform() > 0.5:
                        # hflip
                        image_tensors = image_tensors[:,:,::-1,:]
                        for ii in range(len(gt_boxes)):
                            gt_boxes[ii] = tf.stack([1.-gt_boxes[ii][:,2],gt_boxes[ii][:,1],1.-gt_boxes[ii][:,0],gt_boxes[ii][:,3]],axis=1)
                        
                    if np.random.uniform() > 0.5:
                        # vflip
                        image_tensors = image_tensors[:,::-1,:,:]
                        for ii in range(len(gt_boxes)):
                            gt_boxes[ii] = tf.stack([gt_boxes[ii][:,0],1.-gt_boxes[ii][:,3],gt_boxes[ii][:,2],1.-gt_boxes[ii][:,1]],axis=1)
                # </augmentation>

                # Training step (forward pass + backwards pass)
                prediction_dict, shapes, total_loss = train_step_fn(image_tensors, gt_boxes, gt_classes, update_weights = True)

                if idx % 100 == 0:
                    # write tensorboard summary
                    with writer_train.as_default():
                        tf.summary.scalar("loss",total_loss,step=idx)
                        
                        prediction_dict = detection_model.postprocess(prediction_dict, shapes)
                        vis = viz_utils.draw_bounding_boxes_on_image_tensors(tf.cast(tf.concat(image_tensors,axis=0),tf.uint8),
                                            prediction_dict['detection_boxes'],
                                            prediction_dict['detection_classes'].numpy().astype(np.uint32) + label_id_offset,#prediction_dict['detection_classes'],#prediction_dict['detection_classes'].astype(tf.int32) + label_id_offset,
                                            prediction_dict['detection_scores'],
                                            category_index)
                        vis = tf.cast(vis,tf.float32)
                        tf.summary.image('prediction',vis/255.,step=idx)
                        writer_train.flush()

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
                        prediction_dict, shapes, _loss_test = train_step_fn(image_tensors_test, gt_boxes_test, gt_classes_test, update_weights = False)
                        test_loss = test_loss + _loss_test/num_test_batches
                    test_losses.append(test_loss)

                    # write tensorboard summary
                    with writer_test.as_default():
                        tf.summary.scalar("loss",test_loss,step=idx)

                        prediction_dict = detection_model.postprocess(prediction_dict, shapes)
                        vis = viz_utils.draw_bounding_boxes_on_image_tensors(tf.cast(tf.concat(image_tensors_test,axis=0),tf.uint8),
                                            prediction_dict['detection_boxes'],
                                            prediction_dict['detection_classes'].numpy().astype(np.uint32) + label_id_offset,#prediction_dict['detection_classes'],#prediction_dict['detection_classes'].astype(tf.int32) + label_id_offset,
                                            prediction_dict['detection_scores'],
                                            category_index)
                        vis = tf.cast(vis,tf.float32)
                        tf.summary.image('prediction',vis/255.,step=idx)
                        writer_test.flush()

                    # check for early stopping -> stop training if test loss is increasing
                    if idx>20000 and config['early_stopping'] and len(test_losses) > 3:
                        if test_loss > test_losses[-2] and test_loss > test_losses[-3] and test_loss > test_losses[-4] and min(test_losses[:-1]) < 1.5*test_losses[-1]:
                            early_stopping = True 
                            print('[*] stopping object detection early at step %i, epoch %i, because current test loss %f is higher than previous %f and %f' % (idx, epoch, test_loss, test_losses[-2], test_losses[-3]))
                            ckpt_saver = tf.compat.v2.train.Checkpoint(detection_model=detection_model)
                            ckpt_manager = tf.train.CheckpointManager(ckpt_saver, checkpoint_directory, max_to_keep=5)
                            saved_path = ckpt_manager.save()
                            print('[*] saved object detection model to',checkpoint_directory,'->',saved_path)
                            return detection_model
                idx += 1 


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    parser.add_argument('--minutes',required=False,default=0.0,type=float)
    #parser.add_argument('--thresh_detection',required=False,default=0.5,type=float)
    args = parser.parse_args()

    config = model.get_config(project_id = args.project_id)
    config['project_id'] = args.project_id
    config['video_id'] = args.video_id
    config['minutes'] = args.minutes
    
    finetune(config)
    