"""
    migration tool

    export multiple videos from a given project. 
        pack all data to a new database file and copy all labeled files, but not the actual video
        zip 

    import migration_zip
        unpack zip
        insert zip db data into own database, move files to correct location
"""
import os 
import shutil 
import numpy as np 
import cv2 as cv 
import zipfile
import time 
from multitracker.be import dbconnection

def export_annotation(args):
    print('[*] export',args)
    tstart = time.time()
    project_id = int(args.project_id)
    video_ids = [int(iid) for iid in args.video_ids.split(',')]
    
    # load source database
    db = dbconnection.DatabaseConnection(file_db = os.path.join(args.data_dir, 'data.db'))
    
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
            _, _, frame_idx, x1, y1, x2, y2 = _d
            frame_file = os.path.join( args.data_dir, 'projects', str(project_id), str(video_id), 'frames', 'train', frame_idx + '.png' )
            labeled_files.append( frame_file )

        for _d in keypoints_data[video_id]:
            labid, video_id, frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y = _d
            frame_file = os.path.join( args.data_dir, 'projects', str(project_id), str(video_id), 'frames', 'train', frame_idx + '.png' )
            labeled_files.append( frame_file )
    labeled_files = list(set(labeled_files))

    workdir = '/tmp/multitracker_migrate'
    if os.path.isdir(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)

    ## create new database in output directory and insert all the data 
    db_out = dbconnection.DatabaseConnection(file_db = os.path.join(workdir, 'data.db'))
    # insert new project
    query = "insert into projects (name, manager, keypoint_names, created_at) values(?,?,?,?);"
    new_project_id = db_out.insert(query, project_data[1:])

    # insert new videos
    new_video_ids = []
    for video_id in video_ids:
        query = "insert into videos (name, project_id, fixed_number) values (?,?,?)"
        new_video_id = db_out.insert(query, video_data[video_id][1:])
        new_video_ids.append(new_video_id)

    # insert boxes
    for old_video_id, new_video_id in zip(video_ids, new_video_ids):
        query = "insert into bboxes (video_id, frame_idx, x1, y1, x2, y2) values (?,?,?,?,?,?);"
        for dat in box_data[old_video_id]:
            dat = dat[2:]
            dat = tuple([new_video_id]+list(dat))
            db_out.insert(query, dat)

    # insert keypoints
    for old_video_id, new_video_id in zip(video_ids, new_video_ids):
        query = "insert into keypoint_positions (video_id, frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y) values (?,?,?,?,?,?);"
        for dat in keypoints_data[old_video_id]:
            dat = dat[2:]
            dat = tuple([new_video_id]+list(dat))
            db_out.insert(query, dat)
                    
    ## copy files to output directory
    print('[*] copying %i files' % len(labeled_files))
    for f in labeled_files:
        #fo = os.path.join( workdir, '/projects/' + f.split('/projects/')[1] )
        fo = workdir + '/projects/' + f.split('/projects/')[1]
        if not os.path.isfile(f):
            print('[ERROR] could not find file', f )
        else:
            #print(f,'->',fo)
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True,default=None,help='whether to import or export data')
    parser.add_argument('--data_dir',required=True,help="base data directory that contains 'data.db' and folder 'projects'")
    parser.add_argument('--project_id',required=True,type=int,help='what project to export or what project should import data')
    parser.add_argument('--video_ids',default='',help='video ids to export')
    parser.add_argument('--zip', required=True,default=None,help='file name of zip to export or import')

    args = parser.parse_args()
    assert args.mode in ['import','export']

    if args.mode == 'export':
        export_annotation(args)
    else:
        import_annotation(args)

