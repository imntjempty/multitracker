# Upper Bound Multitracker

This is a framework for tracking animals and their corresponding limbs. It assumes, that the number of objects visible in the video is fixed and known. It uses Faster-RCNN or SSD for object detection and Stacked Hourglasses for keypoint detection. A UpperBound Tracker keeps track of their positions. These mice were tracked with Fixed Multitracker https://www.youtube.com/watch?v=mQenxsiJWBQ. Note the long continous object and keypoint tracks although no frames of this video were provided for any model training.

## Motivation
Multiple Object Tracking (MOT) is defined in an open world: for each frame it is unknown how many objects are currently observed, and therefore can be visible at the same time. Also the total number of objects seen in the complete video is unknown.

In this specific use case, requirements differ slightly. Animals are filmed in enclosured environments in fixed and known numbers. These biological studies expect a limited set of very long tracks, each track corresponding to the complete movement of one animal in parts of the video. Tracking algorithms will produce fragmented tracks that have to be merged manually after the tracking process.

This leads to the definition of Upper Bound Tracking, that tries to track multiple animals with a known upper bound `u ∈ N` of the video `v` as the maximum number
of indivudual animals filmed at the same time. Therefore a new tracking algorithm was developed to improve fragmentation that exploits the upper bound of videos, called Upper Bound Tracker (UBT). It needs, besides the RGB video stream, the upper bound `u ∈ N`. It is inspired by the V-IoU tracker and extended by careful consideration before creating new tracks to never have more than `u` tracks at the same time. Additionally a novel reidentification step is introduced, that updates an inactive and probably lost track to the position of a new detection if `u` tracks are already present. By introducing the upper bound `u`, tracking algorithms can exploit the provided additional knowledge to improve matching or reidentification.

![alt text](https://github.com/dolokov/multitracker/blob/9d8960a2352f33017b728873b4a9f15b710a15c5/app/static/images/UBT_frame_update.png)

## Installation
A dedicated conda environment is recommended. You can set it up as follows:

```
sudo apt install g++ -y && sudo apt install cmake -y && sudo apt install git -y && sudo apt install protobuf-compiler -y
conda create --name multitracker python=3.7
conda activate multitracker
conda install ipython
conda install pip
pip install -r requirements.txt
cd /tmp && git clone https://github.com/tensorflow/models && cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
cd ~ && git clone https://github.com/dolokov/multitracker
conda install cudatoolkit # hotfix for tf bug https://github.com/tensorflow/tensorflow/issues/45930
```
## Getting Started
### Create Project
First create a new project. A project has a name and a set of keypoint names. It can contain multiple videos.

```python3.7 -m multitracker.be.project -name MiceTop -manager MyName -keypoint_names nose,body,left_ear,right_ear,left_front_feet,right_front_feet,left_back_feet,right_back_feet ```

Note, that the keypoint detection uses horizontal and vertical flipping for data augmentation while training, which might violate some label maps. This is automatically fixed by dynamically switching labels of pairs of classes that are simliar expect `left` and `right` in the name. (e.g. `left_ear` and `right_ear` are switched, `l_ear` and `r_ear` are not).

### Add Video
Then add a video to your project with ID 1. It will write every frame of the video to your local disk for later annotation.

```python3.7 -m multitracker.be.video -add_project 1 -add_video /path/to/video.mp4```

### Label Frames
Fixed Multitracker tracks objects and keypoints and therefore offers two annotation tools for drawing bounding boxes and setting predefined, project dependent keypoints. 

```python3.7 -m multitracker.app.app```

Go to the url http://localhost:8888/home . You should see a list of your projects and videos. You then can start each annotation tool with a link for the specific tool and video you want annotate. Please note that you should have an equal number of labeled images for both tasks. We recommend to annotate at least 250 frames, but the more samples the better the detections.

### Track Animals
Now you can call the actual tracking algorithm . If not provided with pretrained models for object detection and keypoint models, it will train those based on annotations of the supplied video ids.

```python3.7 -m multitracker.tracking --project_id 7 --test_video_ids 13,14 --train_video_ids 9,14 --video /path/to/target_video.mp4```

This will create a video showing the results of the tracking process. The default configuration is optimized for highest quality. To speed up calculation, try this low resolution configuration:

```python3.7 -m multitracker.tracking --project_id 7 --train_video_ids 9,14 --test_video_ids 13,14 --objectdetection_resolution 320x320 --objectdetection_method ssd --keypoint_resolution 96x96 --video /path/to/target_video.mp4```

## Advanced Usage
Multitracker is a top-down pipeline, that first uses Google's Object Detection framework to detect and crop all animals, followed by a custom semantic segmentation for keypoint detection on these crops. The tracking method DeepSORT also needs an autoencoder to extract visual features for reidentification. Therefore two or three models are needed for tracking. Multitracker implements a variety of different neural networks for solving object detection and keypoint estimation. 
Trained models can be supplied as command line arguments to avoid retraining and allow easy recombination of different model checkpoints. Trained models can be found in the directory `~/checkpoints/multitracker`. This is also very helpful if multiple videos should be tracked with one set of pretrained models.

### Monitor training progress
Tensorboard summaries and images are logged periodically while training. Start the server and monitor it by opening http://localhost:6006
`tensorboard --logdir ~/checkpoints/multitracker`

### Track with pretrained models
If no paths to the three possible model types are given with arguments `--objectdetection_model`, `--keypoint_model`, `--autoencoder_model`

```python3.7 -m multitracker.tracking --project_id 7 --test_video_ids 13,14 --train_video_ids 9,14 --video /path/to/target_video.mp4 --objectdetection_model /path/to/objdetect --keypoint_model /path/to/keypoint --autoencoder_model /path/to/ae```

### Arguments 


There are several options for object detection, keypoint estimation and tracking. Each combination might give different results and can be easily changed.

`--project_id` ID of the project. Each project has a unique label map.

`--video` path of the MP4 video that should be tracked.

`--train_video_ids` list of video ids that are trained on (eg 1,2,3)

`--test_video_ids` list of video ids that are tested on (eg 3,4)
 
`--data_dir` directory to save all data and database. defaults to ~/data/multitracker

`--tracking_method` options: DeepSORT, VIoU, UpperBound

`--upper_bound` upper bound number of animals observed

`--objectdetection_method` options: fasterrcnn, ssd. fasterrcnn is slower but usually achieves higher accuracy. ssd is faster but might fail to generalize to new videos

`--keypoint_method` options: none, hourglass2, hourglass4, hourglass8, vgg16, efficientnet, efficientnetLarge, psp. defaults to hourglass2. option none tracks objects without keypoints.

`--objectdetection_resolution` resolution used in object detection. defaults to 640x640

`--keypoint_resolution` resolution used in keypoint detection. defaults to 224x224

`--track_tail` length of drawn tail for all animals in visualization

`--delete_all_checkpoints` delete all checkpoints from directory ~/checkpoints/multitracker

Each predicted bounding box and keypoint comes with its own confidence score indicating how sure the algorithm is the object or keypoint to actually be there. We filter these predictions based on two thresholds, that can be changed:

`--min_confidence_boxes` minimum confidence for an detected animal bounding box, defaults to 0.5

`--min_confidence_keypoints` minimum confidence for an detected limb keypoint, defaults to 0.5

## Troubleshooting
### Out Of Memory Errors or very long runtime
- Faster R-CNN is a big model, try the smaller SSD ```--objectdetection_method ssd``` 
- lower the resolution for object detection ```--objectdetection_resolution 320x320```
- lower the resolution for keypoint detection ```--keypoint_resolution 96x96```
- downsample the input video 

### No boxes are detected
- Check out tensorboard images called 'object detection'. If the train predictions look great, but the test predictions are awful, label more bounding boxes!
- Faster R-CNN sometimes fails on very small boxes, try changing the backbone to SSD ```--objectdetection_method ssd``` 
- try to lower the threshold for bounding boxes ```--min_confidence_boxes 0.25```

### No keypoints are detected
- Check out tensorboard images. If the train predictions look great, but the test predictions are awful, label more keypoints and bounding boxes!
- change the backbone ```--keypoint_method psp```
- lower the threshold ```--min_confidence_keypoints 0.25```

## Replicate Experiments
If you want to have access to the utilized data, send a short request to alexander.dolokov at gmail.

