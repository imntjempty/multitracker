"""
    === Video Handler ===
    - add videos to project
        copy video to project dir
        sample all frames, each video different directory
    
    python3.7 -m multitracker.be.video -add_project 0 -add_video /home/alex/data/mice/v1/Superenrichment_Gruppe3_2.mts

"""


import os,sys,time
from glob import glob 
import numpy as np 
import cv2 as cv 
import shutil
import subprocess

#import multitracker
#from multitracker import util 


def add_video_to_project(base_dir, project_id, source_video_file):
    project_dir = os.path.join(base_dir,str(project_id))

    video_dir = os.path.join(project_dir,"videos")
    if not os.path.isdir(video_dir ):
        os.makedirs(video_dir)
    video_file = os.path.join(video_dir,source_video_file.split('/')[-1])

    # insert video into db
    video_id = 0
    #connector.execute("""insert into videos (project_id, video, inserted_at) values (%i, '%s', '%s');""" % (project_id, video_file, multitracker.util.get_now()))


    frames_dir = os.path.join(project_dir,str(video_id),'frames')
    if not os.path.isdir(frames_dir):
        os.makedirs(os.path.join(frames_dir,"train"))
        os.makedirs(os.path.join(frames_dir,"test"))
        #os.makedirs(os.path.join(project_dir,"/data"))

    # copy file to project directory    
    shutil.copy(source_video_file, video_file)

    # sample frames 
    subprocess.call(['ffmpeg','-i',video_file, '-vf', 'fps=30', frames_dir+'/%d.png'])
    

    # split frames into train/test half/half
    frames = sorted(glob(os.path.join(frames_dir, '*.png')))
    num_frames = len(frames)
    split_frame_idx = num_frames // 2
    for i in range(split_frame_idx):
        os.rename(frames[i], os.path.join(frames_dir,'train',frames[i].split('/')[-1]))
    for i in range(split_frame_idx,num_frames):
        os.rename(frames[i], os.path.join(frames_dir,'test',frames[i].split('/')[-1]))

    print('[*] added video %s to project %i with new video id %i.' % (source_video_file, project_id, video_id))

    return video_id 

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-add_project',type=int,required=True)
    parser.add_argument('-add_video',type=str,required=True)
    parser.add_argument('-base_dir',required=False,default = os.path.expanduser('~/data/multitracker'))
    args = parser.parse_args()

    add_video_to_project(args.base_dir, args.add_project, args.add_video)
