"""
    migration tool

    export multiple videos from a given project. 
        pack all data to a new database file and copy all labeled files, but not the actual video
        zip 

    import migration_zip
        unpack zip
        insert zip db data into own database, move files to correct location
"""
from glob import glob 
import os 
import shutil 
import numpy as np 
import cv2 as cv 
import zipfile
import time 
import pickle
from multitracker.be import dbconnection
from multitracker.object_detection import finetune 

def setup_workdir():
    workdir = '/tmp/multitracker_migrate'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    return workdir

def export_annotation(args):
    print('[*] export',args)
    tstart = time.time()
    project_id = int(args.project_id)
    video_ids = [int(iid) for iid in args.video_ids.split(',')]
    
    # load source database
    db = dbconnection.DatabaseConnection(file_db = os.path.join(args.data_dir, 'data.db'))
    
    # hacky: make sure that all the frames are written to disk
    finetune.get_bbox_data({'project_id': project_id, 'data_dir': args.data_dir}, args.video_ids, abort_early = True)

    ## fetch project, video, bounding boxes and keypoints data from source database
    db.execute("select * from projects where id = %i;" % project_id)
    project_data = db.cur.fetchone()
    
    video_data = {}
    box_data = {}
    keypoints_data = {}
    labeled_files = []
    for video_id in video_ids:
        db.execute("select * from videos where id = %i;" % video_id)
        video_data[video_id] = [x for x in db.cur.fetchall()][0]
        
        db.execute('select * from bboxes where video_id = %i;' % video_id)
        box_data[video_id] = [x for x in db.cur.fetchall()]

        db.execute('select * from keypoint_positions where video_id = %i;' % video_id)
        keypoints_data[video_id] = [x for x in db.cur.fetchall()]

        for _d in box_data[video_id]:
            _, _, frame_idx, individual_id, x1, y1, x2, y2, is_visible = _d
            frame_file = os.path.join( args.data_dir, 'projects', str(project_id), str(video_id), 'frames', 'train', '%05d.png' % int(frame_idx) )
            labeled_files.append( frame_file )

        for _d in keypoints_data[video_id]:
            labid, video_id, frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y, is_visible = _d
            frame_file = os.path.join( args.data_dir, 'projects', str(project_id), str(video_id), 'frames', 'train', '%05d.png' % int(frame_idx) )
            labeled_files.append( frame_file )
    labeled_files = list(set(labeled_files))

    workdir = setup_workdir()
    file_pickle = os.path.join(workdir, 'data.pkl')
    pickle.dump([project_data, video_data, box_data, keypoints_data], open(file_pickle, 'wb'))
                    
    ## copy files to output directory
    print('[*] copying %i files' % len(labeled_files))
    for f in labeled_files:
        fo = workdir + '/projects/' + f.split('/projects/')[1]
        dd = os.path.split(fo)[0]
        if not os.path.isdir(dd): os.makedirs(dd)
        shutil.copy(f, fo)

    ## create zip file
    shutil.make_archive( args.zip[:-4], 'zip', workdir )

    ## delete work directory 
    shutil.rmtree(workdir)

    print('[*] export data to %s successful after %i minutes' % (args.zip, int((time.time()-tstart)/60.)))

def import_annotation(args):
    print('[*] import',args)
    tstart = time.time()
    
    workdir = setup_workdir()
    ## extract zip to workdirectory
    shutil.unpack_archive(args.zip, workdir)

    [project_data, video_data, box_data, keypoints_data] = pickle.load( open( os.path.join(workdir, 'data.pkl'), "rb" ) )
    old_project_id = project_data[0]
    old_video_ids = list(video_data.keys())

    ## create new database in output directory and insert all the data 
    db_out = dbconnection.DatabaseConnection(file_db = os.path.join(args.data_dir, 'data.db'))

    # insert new project
    if args.project_id is None:
        query = "insert into projects (name, manager, keypoint_names, created_at) values(?,?,?,?);"
        new_project_id = db_out.insert(query, project_data[1:])
    else:
        new_project_id = int(args.project_id)

    # insert new videos
    new_video_ids = []
    for video_id in old_video_ids:
        query = "insert into videos (name, project_id, fixed_number) values (?,?,?)"
        viddat = video_data[video_id][1:] # cut id
        viddat = tuple([viddat[0]] + [new_project_id] + [viddat[2]])
        new_video_id = db_out.insert(query, viddat)
        new_video_ids.append(new_video_id)

    # insert boxes
    for old_video_id, new_video_id in zip(old_video_ids, new_video_ids):
        query = "insert into bboxes (video_id, frame_idx, individual_id, x1, y1, x2, y2, is_visible) values (?,?,?,?,?,?,?,?);"
        for dat in box_data[old_video_id]:
            dat = dat[2:] # cut id + video_id
            frame_idx = dat[0]
            fin  = os.path.join(workdir,       'projects', str(old_project_id), str(old_video_id), 'frames', 'train', '%05d.png' % int(frame_idx))
            fout = os.path.join(args.data_dir, 'projects', str(new_project_id), str(new_video_id), 'frames', 'train', '%05d.png' % int(frame_idx))
            if not os.path.isdir(os.path.split(fout)[0]): os.makedirs(os.path.split(fout)[0])
            if not os.path.isfile(fout): shutil.copy(fin, fout)
            if len(dat) == 5: 
                # LEGACY: attributes 'individual_id' and 'is_visible' was added recent so add it as true
                frame_idx, x1, y1, x2, y2 = dat 
                individual_id, is_visible = 1, True
                dat = (frame_idx, individual_id, x1, y1, x2, y2, is_visible)
            dat = tuple([new_video_id]+list(dat))
            db_out.insert(query, dat)

    # insert keypoints
    for old_video_id, new_video_id in zip(old_video_ids, new_video_ids):
        query = "insert into keypoint_positions (video_id, frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y, is_visible) values (?,?,?,?,?,?,?);"
        for dat in keypoints_data[old_video_id]:
            dat = dat[2:] # cut id + video_id
            frame_idx = dat[0]
            fin  = os.path.join(workdir,       'projects', str(old_project_id), str(old_video_id), 'frames', 'train', '%05d.png' % int(frame_idx))
            fout = os.path.join(args.data_dir, 'projects', str(new_project_id), str(new_video_id), 'frames', 'train', '%05d.png' % int(frame_idx))
            if not os.path.isdir(os.path.split(fout)[0]): os.makedirs(os.path.split(fout)[0])
            if not os.path.isfile(fout): shutil.copy(fin, fout)
            if len(dat) == 5:
                # LEGACY: added recently is_visible
                frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y = dat 
                is_visible = True 
                dat = (frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y, is_visible)
            dat = tuple([new_video_id]+list(dat))
            db_out.insert(query, dat)

    shutil.rmtree(workdir)
    print('[*] import data from %s to %s successful after %i minutes' % (args.zip, args.data_dir, int((time.time()-tstart)/60.)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,default=None,help='whether to import or export data')
    parser.add_argument('--data_dir',required=False,default=os.path.expanduser('~/data/multitracker'),help="base data directory that contains 'data.db' and folder 'projects'")
    parser.add_argument('--project_id',required=False,type=int,help='what project to export or what project should import data')
    parser.add_argument('--video_ids',default='',help='video ids to export')
    parser.add_argument('--zip', required=True,default=None,help='file name of zip to export or import')

    args = parser.parse_args()
    assert args.mode in ['import','export']

    if args.mode == 'export':
        export_annotation(args)
    else:
        import_annotation(args)

