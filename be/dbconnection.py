import sqlite3
from sqlite3 import Error
import os 
from random import shuffle 

try:
    from flask import g
except:
    print('[*] not using global context for flask!')

class DatabaseConnection(object):
    def __init__(self,file_db = os.path.expanduser("~/data/multitracker/data.db")):
        self.file_db = file_db
        init_db = not os.path.isfile(self.file_db)
        self.conn = sqlite3.connect(self.file_db,check_same_thread=False)
        self.cur = self.conn.cursor()

        if init_db:
            query = """
                create table if not exists projects (id integer primary key autoincrement, name text, manager text, keypoint_names text, created_at text);
                create table if not exists videos (id integer primary key autoincrement, name text, project_id integer);
                create table if not exists keypoint_positions (id integer primary key autoincrement, video_id integer, frame_idx text, keypoint_name text, individual_id integer, keypoint_x real, keypoint_y real);
                create table if not exists frame_jobs (id integer primary key autoincrement, project_id integer, video_id integer, time real, frame_name text);
            """
            self.cur.executescript(query)
    
        self.list_sep = "&$#"

    def execute(self,query):
        self.cur = self.conn.cursor()
        self.cur.execute(query)    

    def commit(self):
        self.conn.commit()

    def insert(self, query, values):
        self.cur.execute(query, values)
        self.conn.commit()
        return  self.get_last_id()

    def get_last_id(self):
        return self.cur.lastrowid

    def get_keypoint_names(self, project_id, split = True):
        q = """select keypoint_names from projects where id = %i;""" % int(project_id)
        self.execute(q)
        rows = self.cur.fetchall()
        if len(rows) == 0:
            raise Exception("[ERROR] no keypoint names found for project %s" % str(project_id))
        x = rows[0][0]
        if split:
            x = x.split(self.list_sep)
        return x 

    def get_project_name(self,project_id):
        q = """select name from projects where id = %i;""" % int(project_id)
        self.execute(q)
        name = [x for x in self.cur.fetchall()][0][0]
        return name 

    def get_video_name(self, video_id):
        q = """select name from videos where id = %i;""" % int(video_id)
        self.execute(q)
        name = [x for x in self.cur.fetchall()][0][0]
        return name 
        
    def get_random_project_video(self, project_id):
        q = "select id from videos where project_id = %i;" % int(project_id)
        self.execute(q)
        video_ids = [x for x in self.cur.fetchall()]
        if len(video_ids) == 0:
            return None 

        shuffle(video_ids)
        return video_ids[0][0]

    def get_labeled_frames(self, project_id):
        self.execute('''select frame_idx 
                            from keypoint_positions 
                            inner join videos on videos.id = keypoint_positions.video_id
                            where videos.project_id = %i;
        ''' % int(project_id))
        return list(set([x for x in self.cur.fetchall()]))

    def get_count_labeled_frames(self, project_id):
        return len(self.get_labeled_frames(project_id))

    def save_labeling(self, data):
        keypoint_names = self.get_keypoint_names(int(data['project_id']))
        num_parts = len(keypoint_names)
        for i, d in enumerate(data['keypoints']):
            x, y = d['x'], d['y']
            if not (x == -100 and y == -100):
                id_ind = d['id_ind']
                keypoint_name = d['keypoint_name']
                query = """ insert into keypoint_positions (video_id, frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y) values (?,?,?,?,?,?); """

                values = (int(data['video_id']), str(data['frame_idx']), keypoint_name, id_ind, x, y)
                self.insert(query, values)
        print('[*] saved labeling data to database for video %i, frame %s.' %(int(data['video_id']),str(data['frame_idx'])))


def get_from_store(_class):
    attr = "_" + _class.__name__
    db = getattr(g, attr, None)
    if db is None:
        db = _class()
        setattr(g, attr, db)
    return db


def get_db() -> DatabaseConnection:
    return get_from_store(DatabaseConnection)


def list_table(table,where_k = None, where_v = None):
    
    c = DatabaseConnection()
    if where_k is None:
        q = "select * from %s;" % table
    else:
        q = "select * from %s where %s = '%s';" % (table, where_k, where_v)
    c.execute(q)
    dat = [ row for row in c.cur.fetchall()]
    print('[*] list of %s: %i counts' % (table,len(dat)))
    for row in dat[:20]:
        print('[*] labeled %i frames' % DatabaseConnection().get_count_labeled_frames(row[0]),row)
        

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-list_projects', action='store_true')
    parser.add_argument('-list_videos', action='store_true')
    parser.add_argument('-list_keypoints', action='store_true')
    
    args = parser.parse_args()

    if args.list_projects:
        projects = list_table('projects')
        
    if args.list_videos:
        list_table('videos')
    if args.list_keypoints:
        list_table('keypoint_positions')
        