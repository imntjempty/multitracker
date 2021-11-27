"""
    convert bounding box annotation to COCO format so that YOLOX can be used

    https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

    ~/github$ python3.7 -m multitracker.object_detection.cvt2COCO --train_video_ids 3 --test_video_ids 1

"""

import os 
import numpy as np 
import cv2 as cv 
import json 
from multitracker.be import dbconnection
import random 
from random import shuffle 
from glob import glob 
from tqdm import tqdm

def cvt2COCO(config):
    abort_early = False
    config['database'] = os.path.expanduser(config['database'])
    config['outdir'] = os.path.expanduser(config['outdir'])

    config['train_video_ids'] = [int(p) for p in config['train_video_ids'].split(',')]
    config['test_video_ids'] = [int(p) for p in config['test_video_ids'].split(',')]
    
    config['imsize'] = [int(s) for s in config['imsize'].split('x')]
    db = dbconnection.DatabaseConnection(file_db=config['database'])

    ## build COCO jsons 
    outdats = {'train2017':{},'test2017':{}}
    for k,dat in outdats.items():
        outdats[k]['info'] = {'url':'github/dolokov/multitracker'}
        outdats[k]['licences'] = [{'id':1,'name':'apache?','url':'letmegooglethatforyou'}]
        outdats[k]['categories'] = [{'id':1,'name':'animal','supercategory':'object'}]
        outdats[k]['images'] = []
        outdats[k]['annotations'] = []
    annot_id = 0 
    for mode, video_ids in [['train2017',config['train_video_ids']],['test2017',config['test_video_ids']]]:
        for video_id in video_ids:
            frame_bboxes = {}
            db.execute("select * from bboxes where video_id=%i and is_visible=true order by id;" % video_id)
            db_boxxes = [x for x in db.cur.fetchall()]
            print('[*] read %i boxes from video %s' % (len(db_boxxes),video_id))

            random.Random(4).shuffle(db_boxxes)
            for dbbox in db_boxxes:
                _, _, frame_idx, individual_id, x1, y1, x2, y2, is_visible = dbbox
                frame_idx = '%08d' % int(frame_idx)
                nsample = [1,10][int('train' in mode)]
                for ii in range(nsample):
                    _frame_idx = '%08d' % int(int(frame_idx)+ii*1e6)
                    _key = '%i_%s' % (video_id, _frame_idx) 
                    if not _key in frame_bboxes:
                        frame_bboxes[_key] = []
                        
                    frame_bboxes[_key].append(np.array([float(z) for z in [y1,x1,y2,x2]]))
            for i, _key in enumerate(frame_bboxes.keys()):
                frame_bboxes[_key] = np.array(frame_bboxes[_key]) 
                video_id, frame_idx = _key.split('_')

            

            ## open video, check if annotated frames are written to disk, if not, write them
            frames_missing_on_disk = []
            for i, _key in enumerate(frame_bboxes.keys()):
                video_id, frame_idx = _key.split('_')
                frame_path = os.path.join(config['outdir'],mode, '%s_%08d.png' % (str(video_id),int(frame_idx)))
                
                outdats[mode]['images'].append({
                    'id': int(frame_idx),
                    'file_name': frame_path.split('/')[-1]
                })

                if not os.path.isfile(frame_path):
                    frames_missing_on_disk.append([video_id, frame_idx, frame_path])

            video_name = db.get_video_name(int(video_id))
            if '\\' in video_name:
                video_name = os.sep.join(video_name.split('\\'))
            if '/' in video_name:
                video_name = os.sep.join(video_name.split('/'))
            video_name = os.path.split(video_name)[1]
            
            '''video_path = os.path.join(os.path.split(config['database'])[0], 'videos', video_name)
            if not os.path.isfile(video_path):
                video_path = os.path.join(config['outdir'], 'videos', os.path.split(video_name)[1])'''
            video_paths = glob(os.path.join(os.path.split(config['database'])[0],'projects','*','videos',video_name)) + glob(os.path.join(os.path.split(config['database'])[0],'projects','*','*','videos',video_name))
            video_path = video_paths[0]


            print('sampling %i frames'% len(frames_missing_on_disk),' from video',video_id, video_name, video_path)
            assert os.path.isfile(video_path), "[*] ERROR: could not find video on disk!"
            video = cv.VideoCapture(video_path)

            width  = int(video.get(cv.CAP_PROP_FRAME_WIDTH))   # float `width`
            height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`
            for i in range(len(outdats[mode]['images'])):
                outdats[mode]['images'][i]['width'] = width
                outdats[mode]['images'][i]['height'] = height

            if len(frames_missing_on_disk) > 0:
                frames_missing_on_disk = sorted(frames_missing_on_disk, key=lambda x: int(x[1]))    
                
                frame_cnt = 0
                frame = 1 
                with tqdm(total=len(frames_missing_on_disk)) as pbar:
                    while len(frames_missing_on_disk) > 0 and frame is not None:
                        next_target_frame = int(frames_missing_on_disk[0][1])
                        #print(frame_cnt,'next_target_frame',next_target_frame)
                        #print('frames_missing_on_disk',frames_missing_on_disk)
                        _, frame = video.read()
                        if frame_cnt == next_target_frame:
                            # write to disk
                            _path = frames_missing_on_disk[0][2]
                            os.makedirs(os.path.split(_path)[0], exist_ok=True)
                            frame = cv.resize(frame, config['imsize'])
                            cv.imwrite(_path, frame)
                            
                            if 0: # vis debug
                                kk = '%s_%s'%(frames_missing_on_disk[0][0],frames_missing_on_disk[0][1])
                                assert kk in frame_bboxes
                                vis = np.uint8(frame)
                                for _d in frame_bboxes[kk]:
                                    color = (0,0,255)
                                    vis = cv.rectangle(vis,(int(_d[1]),int(_d[0])),(int(_d[3]),int(_d[2])),color,3)
                                path_vis = _path.replace('/%s/'%mode,'/%svis/'%mode)
                                os.makedirs(os.path.split(path_vis)[0], exist_ok=True)
                                cv.imwrite(path_vis, vis)
                            #print('[*] writing annotated frame %s' % frames_missing_on_disk[0][2] )
                            frames_missing_on_disk = frames_missing_on_disk[1:]
                            pbar.update(1)
                        frame_cnt += 1
                

            ## write annotations to json
            for i, _key in enumerate(frame_bboxes.keys()):
                video_id, frame_idx = _key.split('_')
                frame_bboxes[_key] = np.array(frame_bboxes[_key]) 
                for j in range(frame_bboxes[_key].shape[0]):
                    x1,y1,x2,y2 = frame_bboxes[_key][j]
                    w,h = x2-x1,y2-y1
                    outdats[mode]['annotations'].append({
                        'id': annot_id,
                        'image_id': int(frame_idx),
                        'bbox': [x1,y1,w,h], # COCO Bounding box: (x-top left, y-top left, width, height)
                        'category_id': 1, # only one class
                        'iscrowd': 0,
                        'area': w*h
                    })
                    annot_id += 1 
            os.makedirs(os.path.join(config['outdir'],'annotations'),exist_ok=True)
            with open(os.path.join(config['outdir'],'annotations','%s.json'%mode),'w') as f:
                json.dump( outdats[mode], f, indent=4)

        if abort_early:
            return True 



if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_video_ids', required=True)
    parser.add_argument('--test_video_ids', required=True)
    parser.add_argument('--outdir',default='~/github/multitracker/object_detection/YOLOX/datasets/multitracker')
    parser.add_argument('--database', default = '~/data/multitracker/data.db')
    parser.add_argument('--imsize', default = '960x540')
    args = vars(parser.parse_args())
    cvt2COCO(args)