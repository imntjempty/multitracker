import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py

from multitracker import util 
from multitracker.keypoint_detection import heatmap_drawing, model 
from multitracker.be import dbconnection
# <network architecture>
from tensorflow.keras.applications.resnet_v2 import preprocess_input


def point_distance(config, p1, p2):
    dist = np.linalg.norm(np.array(p1)/config['img_height']-np.array(p2)/config['img_height'])
    score = 1. / dist 
    score = min(1, score) # max 1
    return score 

