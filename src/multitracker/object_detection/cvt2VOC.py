"""
    convert bounding box annotation to COCO format so that YOLOX can be used

    https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

    ~/github$ python3.7 -m multitracker.object_detection.cvt2VOC --train_video_ids 3 --test_video_ids 1

    fix yolox https://blog.csdn.net/weixin_42166222/article/details/119637797

    voc tutorial https://blog.csdn.net/nan355655600/article/details/119519294?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.no_search_link&spm=1001.2101.3001.4242.1


    train 
        $python3.7 -m tools.train -f exps/example/yolox_voc/yolox_voc_s.py -d 0 -b 16 --fp16 -c /home/alex/data/multitracker/object_detection/yolox_s.pth

"""

from shutil import copyfile
import os 
import numpy as np 
import cv2 as cv 
import json 
from multitracker.be import dbconnection
import random 
from random import shuffle 
from glob import glob 
from tqdm import tqdm
from copy import deepcopy

def write_voc_xmls(config, mode, frame_bboxes, outdats):
    # https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5

    for _key in frame_bboxes.keys():
        video_id, frame_idx = _key.split('_')
        xml_path = os.path.join(config['outdir'],'Annotations','%s.xml' % _key)
        folder, filename = os.path.split(xml_path)
        folder = 'datasets'+folder.split('/datasets')[1]
        filename = filename.replace('.xml','.jpg').replace('Annotations','JPEGImages')

        width = int(outdats[mode]['images'][0]['width']) 
        height = int(outdats[mode]['images'][0]['height']) 

        str_xml_objects = ""
        for [x,y,w,h] in frame_bboxes[_key]:
            str_xml_objects += '''
                <object>
                    <name>animal</name>
                    <pose>none</pose>
                    <truncated>0</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>%i</xmin>
                        <ymin>%i</ymin>
                        <xmax>%i</xmax>
                        <ymax>%i</ymax>
                    </bndbox>
                </object>
            ''' % (x,y,x+w,y+h)  #(x+w//2,y+h//2,x+w+w//2,y+h+h//2)  
              

        os.makedirs(os.path.split(xml_path)[0],exist_ok=True)
        if os.path.isfile(xml_path): os.remove(xml_path)
        
        with open(xml_path,'w') as f:
            f.write('''
            <annotation>
                <folder>%s</folder>
                <filename>%s</filename>
                <path>./%s/%s</path>
                <source><database>multitracker</database></source>
                <size>
                    <width>%i</width>
                    <height>%i</height>
                    <depth>3</depth>
                </size>
                <segmented>0</segmented>
                %s
            </annotation>
            ''' % (folder,filename,folder,filename,width,height,str_xml_objects) )

def get_txt_split_path(config, mode):
    return os.path.join(config['outdir'],'ImageSets','Main','%s.txt' % mode)

def write_voc_traintest_split(config, mode, frame_bboxes):
    txt_path = get_txt_split_path(config, mode)
    os.makedirs(os.path.split(txt_path)[0],exist_ok=True)
    #if os.path.isfile(txt_path): os.remove(txt_path)
    with open(txt_path,'a+') as f:
        for i, _key in enumerate(frame_bboxes.keys()):
            video_id, frame_idx = _key.split('_')
            frame_path = os.path.join(config['outdir'], 'JPEGImages', '%s_%09d.jpg' % (str(video_id),int(frame_idx)))
            _name = frame_path.split('.jpg')[0]
            _name = _name.split('/')[-1]
            f.write('%s\n' % _name)    
    if 'test' in mode:
        copyfile(txt_path, os.path.join(os.path.split(txt_path)[0],'e.txt'))

def cvt2VOC(config):
    abort_early = False
    config['database'] = os.path.expanduser(config['database'])
    config['outdir'] = os.path.expanduser(config['outdir'])

    config['train_video_ids'] = [int(p) for p in config['train_video_ids'].split(',')]
    config['test_video_ids'] = [int(p) for p in config['test_video_ids'].split(',')]
    
    config['imsize'] = [int(s) for s in config['imsize'].split('x')]
    db = dbconnection.DatabaseConnection(file_db=config['database'])

    for k in ['train2017','test2017']:
        try:
            os.remove( get_txt_split_path(config, k) )
        except:
            pass 

    ## build COCO jsons 
    outdats = {'train2017':{},'test2017':{}}
    for k,dat in outdats.items():
        outdats[k]['info'] = {'year':2021,'version':0.1,'description':'For object detection','date_created':2020}
        outdats[k]['licenses'] = [{'id':1,'name':'GNU General Public License v3.0','url':'https://github.com/zhiqwang/yolov5-rt-stack/blob/master/LICENSE'}]
        outdats[k]['categories'] = [
            {'id':1,'name':'0','supercategory':'0'},
            {'id':2,'name':'1','supercategory':'1'}
        ]
        outdats[k]['type'] = 'instances'
        outdats[k]['images'] = []
        outdats[k]['annotations'] = []
    
    cnt_images = 0 
    image_mapping = {} # map from image to id 
    annot_id = 0 
    for mode, video_ids in [['train2017',config['train_video_ids']],['test2017',config['test_video_ids']]]:
        for video_id in video_ids:
            frame_bboxes = {}
            db.execute("select * from bboxes where video_id=%i and is_visible=true order by id;" % video_id)
            db_boxxes = [x for x in db.cur.fetchall()]
            print('[*] read %i boxes from video %s' % (len(db_boxxes),video_id))

            #random.Random(4).shuffle(db_boxxes)
            for dbbox in db_boxxes:
                _, _, frame_idx, individual_id, x1, y1, x2, y2, is_visible = dbbox
                x1,y1,x2,y2 = [int(z) for z in [x1,y1,x2,y2]]
                
                w, h = x2-x1, y2-y1 
                # correct boxes to have positive width and height 
                if w < 0:
                    x1 += w 
                    w = abs(w)
                if h < 0:
                    y1 += h 
                    h = abs(h)

                frame_idx = '%09d' % int(frame_idx)
                
                _key = '%i_%s' % (video_id, frame_idx) 
                if not _key in frame_bboxes:
                    frame_bboxes[_key] = []
            
                frame_bboxes[_key].append(np.array([float(z) for z in [x1,y1,w,h]]))
            
            ## open video, check if annotated frames are written to disk, if not, write them
            frames_missing_on_disk = []
            for i, _key in enumerate(frame_bboxes.keys()):
                video_id, frame_idx = _key.split('_')
                frame_path = os.path.join(config['outdir'], 'JPEGImages', '%s_%09d.jpg' % (str(video_id),int(frame_idx)))
                
                image_mapping[(int(video_id),int(frame_idx))] = cnt_images
                outdats[mode]['images'].append({
                    'id': cnt_images,
                    'file_name': frame_path.split('/')[-1]
                })
                cnt_images += 1 

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
            video_path = None 
            video_paths = glob(os.path.join(os.path.split(config['database'])[0],'projects','*','videos',video_name)) + glob(os.path.join(os.path.split(config['database'])[0],'projects','*','*','videos',video_name))
            if len(video_paths) > 0:
                video_path = video_paths[0]


                print('sampling %i frames'% len(frames_missing_on_disk),' from video',video_id, video_name, video_path)
                assert os.path.isfile(video_path), "[*] ERROR: could not find video on disk!"
                video = cv.VideoCapture(video_path)

                video_width  = int(video.get(cv.CAP_PROP_FRAME_WIDTH))   # float `width`
                video_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))  # float `height`

                # scale boxes to imsize
                for i, _key in enumerate(frame_bboxes.keys()):
                    sx,sy = config['imsize'][0]/video_width,config['imsize'][1]/video_height
                    frame_bboxes[_key] = np.array(frame_bboxes[_key]) * np.array([sx,sy,sx,sy])
                    
            for i in range(len(outdats[mode]['images'])):
                outdats[mode]['images'][i]['width'] = config['imsize'][0]
                outdats[mode]['images'][i]['height'] = config['imsize'][1]
                outdats[mode]['images'][i]['date_captured'] = '2021'

            

            if len(frames_missing_on_disk) > 0:
                frames_missing_on_disk = sorted(frames_missing_on_disk, key=lambda x: int(x[1]))    
                
                if video_path is None:
                    # find train/test images
                    _frames = {}
                    for fpath in glob(os.path.join(os.path.split(config['database'])[0],'projects',str(config['project_id']),str(video_id),'frames','train','*.png')):
                        path_frame_idx = int(os.path.split(fpath)[1][:-4])
                        _frames[path_frame_idx] = fpath

                frame_cnt = 0
                frame = 1 
                with tqdm(total=len(frames_missing_on_disk)) as pbar:
                    while len(frames_missing_on_disk) > 0 and frame is not None:
                        next_target_frame = int(frames_missing_on_disk[0][1])
                        if video_path is not None:
                            video.set(1, next_target_frame)
                            _, frame = video.read()
                        else:
                            _video_id, frame_idx, frame_path = frames_missing_on_disk[0]
                            int_frame_idx = int(frame_idx)
                            if int_frame_idx not in _frames:
                                frame = 1 
                                print('[*] did not found frame_idx', frame_idx,'in disk frames',list(_frames.keys()))
                            else:
                                frame = cv.imread(_frames[int_frame_idx])
                                # scale boxes to imsize
                                sx,sy = config['imsize'][0]/frame.shape[1],config['imsize'][1]/frame.shape[0]
                                _key = '%i_%09d' % (int(video_id), int(frames_missing_on_disk[0][1]))
                                frame_bboxes[_key] = np.array(frame_bboxes[_key]) * np.array([sx,sy,sx,sy])
                                    
                                #print(frame.shape,'video_id',video_id, 'frame_idx',frame_idx, 'frame_path',frame_path,os.path.isfile(frame_path))

                        if not type(frame) == int:
                            # write to disk
                            _path = frames_missing_on_disk[0][2]
                            os.makedirs(os.path.split(_path)[0], exist_ok=True)
                            frame = cv.resize(frame, config['imsize'])
                            cv.imwrite(_path, frame)
                        else:
                            _key = '%i_%09d' % (int(video_id), int(frames_missing_on_disk[0][1]))
                            del frame_bboxes[_key]
                            print('[*] deleted frame bbox', _key)

                        if 0: # vis debug
                            kk = '%s_%s'%(frames_missing_on_disk[0][0],frames_missing_on_disk[0][1])
                            assert kk in frame_bboxes
                            vis = np.uint8(frame)
                            for _d in frame_bboxes[kk]:
                                color = (0,0,255)
                                px,py,pw,ph = [int(c) for c in _d]
                                vis = cv.rectangle(vis, (px,py),(px+pw,py+ph),color,3)
                                #vis = cv.rectangle(vis, (py,px),(py+ph,px+pw),color,3)
                                
                            path_vis = _path.replace('/JPEGImages/','/JPEGImagesvis/')
                            os.makedirs(os.path.split(path_vis)[0], exist_ok=True)
                            cv.imwrite(path_vis, vis)
                            print('[*] wrote', path_vis)
                        
                        #print('[*] writing annotated frame %s' % frames_missing_on_disk[0][2] )
                        frames_missing_on_disk = frames_missing_on_disk[1:]
                        pbar.update(1)
                        frame_cnt += 1
                    
            ## write VOC xml
            write_voc_xmls(config, mode, frame_bboxes, outdats)
            write_voc_traintest_split(config, mode, frame_bboxes)

        if abort_early:
            return True 



if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', default=1, type=int)
    parser.add_argument('--train_video_ids', required=True)
    parser.add_argument('--test_video_ids', required=True)
    parser.add_argument('--outdir',default='~/github/multitracker/object_detection/YOLOX/datasets/multitracker')
    parser.add_argument('--database', default = '~/data/multitracker/data.db')
    parser.add_argument('--imsize', default = ['1920x1080', '960x540'][0])
    args = vars(parser.parse_args())
    cvt2VOC(args)