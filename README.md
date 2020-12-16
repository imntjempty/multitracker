# Upper Bound Multitracker

This is a framework for tracking animals and their corresponding limbs. It assumes, that the number of objects visible in the video is fixed and known. It uses Faster-RCNN or SSD for object detection and Stacked Hourglasses for keypoint detection. A UpperBound Tracker keeps track of their positions. These mice were tracked with Fixed Multitracker https://www.youtube.com/watch?v=mQenxsiJWBQ

## Installation

## Getting Started
### Create Project
First create a new project. A project has a name and a set of keypoint names. It can contain multiple videos.

```python3.7 -m multitracker.be.project -name MiceTop -manager MyName -keypoint_names nose,body,left_ear,right_ear,left_front_feet,right_front_feet,left_back_feet,right_back_feet ```

### Add Video
Then add a video to your project with ID 1. It will write every frame of the video to your local disk for later annotation.

```python3.7 -m multitracker.be.video -add_project 1 -add_video /path/to/video.mp4```

### Label Frames
Fixed Multitracker tracks objects and keypoints and therefore offers two annotation tools for drawing bounding boxes and setting predefined, project dependent keypoints. 

```python3.7 -m multitracker.app.app```

Go to the url http://localhost:8888/home . You should see a list of your projects and videos. You then can start each annotation tool with a link for the specific tool and video you want annotate. Please note that you should have an equal number of labeled images for both tasks. We recommend to annotate at least 250 frames, but the more samples the better the detections.

### Track Animals
Now you can call the actual tracking algorithm . If not provided with pretrained models for object detection and keypoint models, it will train those based on annotations of the supplied video ids.

```python3.7 -m multitracker.tracking --project_id 7 --video_id 13 --train_video_ids 9,14 --video /path/to/target_video.mp4```

This will create a video showing the results of the tracking process.

## Advanced Usage
Multitracker is a top-down pipeline, that first uses Google's Object Detection framework to detect and crop all animals, followed by a custom semantic segmentation for keypoint detection on these crops. The tracking method DeepSORT also needs an autoencoder to extract visual features for reidentification. Therefore two or three models are needed for tracking. Multitracker implements a variety of different neural networks for solving object detection and keypoint estimation. 
Trained models can be supplied as command line arguments to avoid retraining and allow easy recombination of different model checkpoints. Trained models can be found in the directory `~/checkpoints/multitracker`. This is also very helpful if multiple videos should be tracked with one set of pretrained models.

### Monitor training progress
Tensorboard summaries and images are logged periodically while training. Start the server and monitor it by opening http://localhost:6006
`tensorboard --logdir ~/checkpoints/multitracker`

### Track with pretrained models
If no paths to the three possible model types are given with arguments `--objectdetection_model`, `--keypoint_model`, `--autoencoder_model`

```python3.7 -m multitracker.tracking --project_id 7 --video_id 13 --train_video_ids 9,14 --video /path/to/target_video.mp4 --objectdetection_model /path/to/objdetect --keypoint_model /path/to/keypoint --autoencoder_model /path/to/ae```

### Arguments to choose tracking receipe
There are several options for object detection, keypoint estimation and tracking. Each combination might give different results and can be easily changed.

`--objectdetection_method` options: fasterrcnn, ssd. fasterrcnn is slower but usually achieves higher accuracy. ssd is faster but might fail to generalize to new videos

`--keypoint_method` options: none, hourglass2, hourglass4, hourglass8, vgg16, efficientnet, efficientnetLarge, psp. defaults to hourglass2. option none tracks objects without keypoints.

`--tracking_method` options: DeepSORT, VIoU, UpperBound

Each predicted bounding box and keypoint comes with its own confidence score indicating how sure the algorithm is the object or keypoint to actually be there. We filter these predictions based on two thresholds, that can be changed:

`--min_confidence_boxes` minimum confidence for an detected animal bounding box, defaults to 0.5

`--min_confidence_keypoints` minimum confidence for an detected limb keypoint, defaults to 0.5

## Troubleshooting
### No boxes are detected
- Check out tensorboard images called 'object detection'. If the train predictions look great, but the test predictions are awful, label more bounding boxes!
- Faster R-CNN sometimes fails on very small boxes, try changing the backbone to SSD ```--objectdetection_method ssd``` 
- try to lower the threshold for bounding boxes ```--min_confidence_boxes 0.25```

### No keypoints are detected
- Check out tensorboard images. If the train predictions look great, but the test predictions are awful, label more keypoints and bounding boxes!
- change the backbone ```--keypoint_method psp```
- lower the threshold ```--min_confidence_keypoints 0.25```
