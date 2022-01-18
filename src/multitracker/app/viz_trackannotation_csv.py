"""
    make viz video from csv trackannotation and video
"""

import os 
import numpy as np 
import cv2 as cv 
import argparse 
from multitracker import util
from tqdm import tqdm

def trackannotation_csv2video(csv_filepath, video_in, video_out):
    # read csv
    data = {}
    with open(csv_filepath , 'r') as f:
        lines = [l.replace('\n','').split(',') for l in f.readlines()]
        header = lines[0] # frame_idx	idv	x1	y1	x2	y2
        lines = lines[1:] # skip header
    for ls in lines:
        if not int(ls[0]) in data:
            data[int(ls[0])] = {}
        data[int(ls[0])][int(ls[1])] = {
            'idv': int(ls[1]),
            'x1': float(ls[2]), 'y1': float(ls[3]), 'x2': float(ls[4]), 'y2': float(ls[5])
        }
    frame_idxs = sorted(list(data.keys()))

    # open input video
    vs_in = cv.VideoCapture(video_in)

    # open output video
    if video_out is not None:
        if os.path.isfile(video_out): os.remove(video_out)
        import skvideo.io
        video_writer = skvideo.io.FFmpegWriter(video_out, outputdict={
            '-vcodec': 'libx264',  #use the h.264 codec
            '-crf': '0',           #set the constant rate factor to 0, which is lossless
            '-preset':'veryslow'   #the slower the better compression, in princple, try 
                                    #other options see https://trac.ffmpeg.org/wiki/Encode/H.264
        }) 

    colors = util.get_colors()
    running = True 
    frame_cnt = 1
    
    with tqdm(total=len(list(data.keys()))) as pbar:
        while running:
            _, frame = vs_in.read()
            if frame is None or len(list(data.keys())) == 0: # stop if video or csv end
                running = False 
            vis = np.array(frame, copy=True)
            if frame_cnt in data:
                for idv, _d in data[frame_cnt].items():
                    color = [int(c) for c in colors[idv]][::-1]
                    vis = cv.rectangle(vis,(int(_d['x1']),int(_d['y1'])),(int(_d['x2']),int(_d['y2'])),color,3)
                    vis = cv.putText( vis, str(idv), (int(_d['x1'])+5,int(_d['y1'])+25), cv.FONT_HERSHEY_COMPLEX, 0.75, color, 2 )
                del data[frame_cnt]

            if video_out is not None:
                vis = cv.putText( vis, str(frame_cnt), (20,40), cv.FONT_HERSHEY_COMPLEX, 1.15, [255,0,0], 2 )
                video_writer.writeFrame(cv.cvtColor(vis, cv.COLOR_BGR2RGB))
            
            cv.imshow('viz', vis)
            cv.waitKey(4*30)
            frame_cnt += 1 
            pbar.update(1) 


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video_in')
    p.add_argument('--video_out')
    p.add_argument('--csv')
    args = p.parse_args()
    trackannotation_csv2video(args.csv, args.video_in, args.video_out)
