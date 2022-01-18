import sqlite3
from sqlite3 import Error
import os 
from random import shuffle 

try:
    from flask import g
except:
    print('[*] not using global context for flask!')

base_data_dir = os.path.expanduser('~/data/multitracker') #os.getenv('MULTITRACKER_DATA_DIR') #
if base_data_dir is None:
    ## ask the user where to put the data dir
    print(3*'\n',10* '*','\n*\n*   The data directory is not set yet. where would you like to put the database and files? (default ~/data/multitracker)')
    base_data_dir = os.path.expanduser(input("Please enter the data directory:"))
    os.environ['MULTITRACKER_DATA_DIR'] = base_data_dir
    print('[*] set data directory to', base_data_dir)

class DatabaseConnection(object):
    def __init__(self,file_db = os.path.join(base_data_dir, "data.db")):
        self.file_db = file_db
        init_db = not os.path.isfile(self.file_db)
        if not os.path.isdir(os.path.split(self.file_db)[0]): os.makedirs(os.path.split(self.file_db)[0])
        try:
            self.conn = sqlite3.connect(self.file_db,check_same_thread=False)
        except:
            raise Exception("   Error! Can not open db file %s" % self.file_db)

        self.cur = self.conn.cursor()

        if init_db:
            query = """
                create table if not exists projects (id integer primary key autoincrement, name text, manager text, keypoint_names text, created_at text);
                create table if not exists videos (id integer primary key autoincrement, name text, project_id integer, fixed_number integer);
                create table if not exists keypoint_positions (id integer primary key autoincrement, video_id integer, frame_idx text, keypoint_name text, individual_id integer, keypoint_x real, keypoint_y real, is_visible bool);
                create table if not exists bboxes (id integer primary key autoincrement, video_id integer, frame_idx text, individual_id integer, x1 real, y1 real, x2 real, y2 real, is_visible bool);
                create table if not exists frame_jobs (id integer primary key autoincrement, project_id integer, video_id integer, time real, frame_name text);
                create table if not exists trackdata (id integer primary key autoincrement, project_id integer, video_id integer, frame_id integer, individual_id integer, x1 real, y1 real, x2 real, y2 real, updated_by text);
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
        name = name.split('/')[-1]
        return name 

    def get_video_fixednumber(self, video_id):
        q = """select fixed_number from videos where id = %i;""" % int(video_id)
        self.execute(q)
        return [x for x in self.cur.fetchall()][0][0]
        
    def get_random_project_video(self, project_id):
        q = "select id from videos where project_id = %i;" % int(project_id)
        self.execute(q)
        video_ids = [x for x in self.cur.fetchall()]
        if len(video_ids) == 0:
            return None 

        shuffle(video_ids)
        return video_ids[0][0]

    def get_all_labeled_frames(self):
        self.execute('''select video_id, frame_idx from keypoint_positions ''')
        return list(set([x for x in self.cur.fetchall()]))


    def get_labeled_frames(self, video_id):
        self.execute('''select frame_idx 
                            from keypoint_positions 
                            where video_id = %i;''' % int(video_id))
        return list(set([x[0] for x in self.cur.fetchall()]))

    def get_count_labeled_frames(self, video_id):
        return len(self.get_labeled_frames(video_id))

    def get_count_all_labeled_frames(self):
        dd = self.get_all_labeled_frames()
        counts = {}
        for [video_id, frame_idx] in dd:
            if not video_id in counts:
                counts[video_id] = 0 
            counts[video_id] += 1 
        return counts 


    def get_labeled_bbox_frames(self, video_id):
        self.execute('''select frame_idx 
                            from bboxes
                            where video_id = %i''' % int(video_id))
        return list(set([x[0] for x in self.cur.fetchall()]))

    def get_count_labeled_bbox_frames(self, video_id):
        return len(self.get_labeled_bbox_frames(video_id))

    def get_all_labeled_bbox_frames(self):
        self.execute('''select video_id, frame_idx from bboxes''')
        return list(set([x for x in self.cur.fetchall()]))

    def get_count_all_labeled_bbox_frames(self):
        dd = self.get_all_labeled_bbox_frames()
        counts = {}
        for [video_id, frame_idx] in dd:
            if not video_id in counts:
                counts[video_id] = 0 
            counts[video_id] += 1 
        return counts 

    def get_frame_annotations(self, video_id, frame_id):
        # find all objects and keypoints
        animals_dat = {}
        query = """
            select bboxes.id as box_id, bboxes.individual_id, bboxes.x1, bboxes.y1, bboxes.x2, bboxes.y2, bboxes.is_visible as box_is_visible, kp.id as keypoint_id, kp.keypoint_name, kp.keypoint_x, kp.keypoint_y, kp.is_visible as keypoint_is_visible 
                from bboxes 
                inner join keypoint_positions as kp 
                on bboxes.video_id=kp.video_id and bboxes.frame_idx=kp.frame_idx and bboxes.individual_id=kp.individual_id 
                where bboxes.video_id=%i and bboxes.frame_idx=%i
                order by bboxes.id, kp.id ASC;
        """ % (video_id, frame_id)
        
        self.execute(query)
        #lres = list(set([x for x in self.cur.fetchall()]))#[0]
        lres = list([x for x in self.cur.fetchall()])#[0]
        #print(query, lres)
        if len(lres) == 0:
            return None 
        
        for line in lres:
            klist = ['box_id', 'individual_id', 'x1', 'y1', 'x2', 'y2', 'box_is_visible', 'keypoint_id', 'keypoint_name', 'keypoint_x', 'keypoint_y', 'keypoint_is_visible']
            d = {}
            #for k, v in zip(klist, line.replace(' ','').split(',')):
            #print(line)
            for k, v in zip(klist, list(line)):
                d[k] = v 
            
            d['keypoint_x'], d['keypoint_y'] = int(round(d['keypoint_x'])), int(round(d['keypoint_y']))

            if not d['box_id'] in animals_dat:
                animals_dat[d['box_id']] = {
                    'is_visible': d['box_is_visible'],
                    'box': [d['x1'],d['y1'],d['x2'],d['y2']],
                    'id': str(d['individual_id']),
                    'keypoints_dat': {},
                    'db_id': d['box_id']
                }
            if d['keypoint_name'] not in animals_dat[d['box_id']]['keypoints_dat']:
                animals_dat[d['box_id']]['keypoints_dat'][d['keypoint_name']] = {
                    'db_id': d['keypoint_id'],
                    'id': str(d['individual_id']),
                    'x': d['keypoint_x'],
                    'y': d['keypoint_y'],
                    'name': d['keypoint_name'],
                    'is_visible': d['keypoint_is_visible']
                }
                animals_dat[d['box_id']]['keypoints'] = list(animals_dat[d['box_id']]['keypoints_dat'].values())
        animals_list = list(animals_dat.values())
        return animals_list

    def save_keypoint_labeling(self, data):
        keypoint_names = self.get_keypoint_names(int(data['project_id']))
        num_parts = len(keypoint_names)
        
        ## TODO: validate input before deleting

        ## don't allow multiple versions of the same frame, if one already in db, delete it first
        self.execute(""" delete from keypoint_positions where video_id=%i and frame_idx=%s;""" % (int(data['video_id']), str(data['frame_idx'])))
        # self.commit()

        for i, d in enumerate(data['keypoints']):
            x, y = d['x'], d['y']
            if not (x == -100 and y == -100):
                id_ind = d['id_ind']
                keypoint_name = d['keypoint_name']
                query = """ insert into keypoint_positions (video_id, frame_idx, keypoint_name, individual_id, keypoint_x, keypoint_y, is_visible) values (?,?,?,?,?,?,?); """
                is_visible = d['is_visible'] 
                values = (int(data['video_id']), str(data['frame_idx']), keypoint_name, id_ind, x, y, is_visible)
                self.insert(query, values)
        #print(data)
        print('[*] saved keypoint labeling data to database for video %i, frame %s.' %(int(data['video_id']),str(data['frame_idx'])))

    def save_bbox_labeling(self, data):
        self.execute(""" delete from bboxes where video_id=%i and frame_idx=%s;""" % (int(data['video_id']), str(data['frame_idx'])))
        
        
        for i, bbox in enumerate(data['bboxes']):
            is_visible = bbox['is_visible'] 
            query = """ insert into bboxes (video_id, frame_idx, individual_id, x1, y1, x2, y2, is_visible) values (?,?,?,?,?,?,?,?); """
            values = (int(data['video_id']), str(data['frame_idx']), bbox['id_ind'], bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],is_visible)
            self.insert(query, values)
        #print(data)
        print('[*] saved bbox labeling data to database for video %i, frame %s.' %(int(data['video_id']),str(data['frame_idx'])))


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
        
def list_frames():
    c = DatabaseConnection()
    q = "select distinct frame_idx from bboxes;"
    c.execute(q)
    dat = [ row for row in c.cur.fetchall()]
    for i,row in enumerate(dat):
        print('[* %i/%i] labeled frame %s' % (i,len(dat),row[0]))

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-list_projects', action='store_true')
    parser.add_argument('-list_videos', action='store_true')
    parser.add_argument('-list_keypoints', action='store_true')
    parser.add_argument('-list_frames', action='store_true')
    
    args = parser.parse_args()

    if args.list_projects:
        projects = list_table('projects')
        
    if args.list_videos:
        list_table('videos')
    if args.list_keypoints:
        list_table('keypoint_positions')
    if args.list_frames:
        list_frames()
        