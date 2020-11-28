# Upper Bound Multitracker

This is a framework for tracking animals and their corresponding limbs. It assumes, that the number of objects visible in the video is fixed and known. It uses Faster-RCNN or SSD for object detection and Stacked Hourglasses for keypoint detection. A FixedAssigner Tracker keeps track of their positions. These mice were tracked with Fixed Multitracker https://www.youtube.com/watch?v=mQenxsiJWBQ

## Installation

## Getting Started
### Create Project
First create a new project. A project has a name and a set of keypoint names. It can contain multiple videos.

```python3.7 -m multitracker.be.project -name Lama1 -manager MyName -keypoint_names nose,body,left_ear,right_ear,left_front_feet,right_front_feet,left_back_feet,right_back_feet ```

### Add Video
Then add a video to your project with ID 1. It will write every frame of the video to your local disk for later annotation.

```python3.7 -m multitracker.be.video -add_project 1 -add_video /path/to/video.mp4```

### Label Frames
Fixed Multitracker tracks objects and keypoints and therefore offers two annotation tools for drawing bounding boxes and setting predefined, project dependent keypoints. 

```python3.7 -m multitracker.app.app```

Go to the url http://localhost:8888/home . You should see a list of your projects and videos. You then can start each annotation tool with a link for the specific tool and video you want annotate. Please note that you should have an equal number of labeled images for both tasks. We recommend to annotate at least 250 frames, but the more samples the better the detections.

### Track Animals
Now you can call the actual tracking algorithm . If not provided with pretrained models for object detection and keypoint models, it will train those based on annotations of the supplied video ids.

```python3.7 -m multitracker.tracking --project_id 7 --video_id 13 --train_video_ids 9,14 --min_confidence_boxes 0.7 --min_confidence_keypoints 0.5 --tracking_method FixedAssigner --video /path/to/target_video.mp4```

This will create a video showing the results of the tracking process.
