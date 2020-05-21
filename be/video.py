"""
    === Video Handler ===
    - add videos to project
        copy video to project dir
        sample all frames, each video different directory
    
    python3.7 -m multitracker.be.video -add_project 1 -add_video /home/alex/data/mice/v1/Superenrichment_Gruppe3_2.mts

"""


import os,sys,time
from glob import glob 
import numpy as np 
import cv2 as cv 
import shutil
import subprocess

#import multitracker
#from multitracker import util 
from multitracker.be import dbconnection

base_dir_default = os.path.expanduser('~/data/multitracker/projects')

def get_frames_dir(project_dir, video_id):
    return os.path.join(project_dir,str(video_id),'frames')

def get_project_dir(base_dir, project_id):
    return os.path.join(base_dir,str(project_id))

def add_video_to_project(base_dir, project_id, source_video_file):
    project_dir = get_project_dir(base_dir, project_id)

    video_dir = os.path.join(project_dir,"videos")
    if not os.path.isdir(video_dir ):
        os.makedirs(video_dir)
    video_file = os.path.join(video_dir,source_video_file.split('/')[-1])

    # save video to db
    query = "insert into videos (name, project_id) values (?,?)"
    conn = dbconnection.DatabaseConnection()
    video_id = conn.insert(query, (video_file, int(project_id)) )

    frames_dir = get_frames_dir(project_dir, video_id)
    if not os.path.isdir(frames_dir):
        os.makedirs(os.path.join(frames_dir,"train"))
        os.makedirs(os.path.join(frames_dir,"test"))
        #os.makedirs(os.path.join(project_dir,"/data"))

    # copy file to project directory    
    if not os.path.isfile(video_file):
        shutil.copy(source_video_file, video_file)

    # sample frames 
    subprocess.call(['ffmpeg','-i',video_file, '-vf', 'fps=30','-vf', "scale=iw/2:ih/2", frames_dir+'/%05d.png'])
    

    # split frames into train/test half/half
    frames = sorted(glob(os.path.join(frames_dir, '*.png')))
    num_frames = len(frames)
    split_frame_idx = num_frames // 2
    for i in range(split_frame_idx):
        os.rename(frames[i], os.path.join(frames_dir,'train',frames[i].split('/')[-1]))
        os.rename(frames[i+split_frame_idx], os.path.join(frames_dir,'test',frames[i+split_frame_idx].split('/')[-1]))

    print('[*] added video %s to project %i with new video id %i.' % (source_video_file, project_id, video_id))

    return video_id 

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-add_project',type=int,required=True)
    parser.add_argument('-add_video',type=str,required=True)
    parser.add_argument('-base_dir',required=False,default = base_dir_default)
    args = parser.parse_args()

    add_video_to_project(args.base_dir, args.add_project, args.add_video)
