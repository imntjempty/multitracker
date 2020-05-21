import sqlite3
from sqlite3 import Error
import os 

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
            """
            self.cur.executescript(query)
    
        self.list_sep = "&$#"

    def execute(self,query):
        self.cur = self.conn.cursor()
        self.cur.execute(query)    

    def commit(self):
        self.conn.commit()

    def get_last_id(self):
        return self.cur.lastrowid

    def get_keypoint_names(self, project_id):
        q = """select keypoint_names from projects where id = %i;""" % int(project_id)
        self.execute(q)
        rows = self.cur.fetchall()
        if len(rows) == 0:
            raise Exception("[ERROR] no keypoint names found for project %s" % str(project_id))
        return rows[0][0]

def list_table(table):
    print('[*] list of %s:' % table)
    c = DatabaseConnection()
    q = "select * from %s;" % table
    c.execute(q)
    for row in c.cur.fetchall():
        print(row)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-list_projects', action='store_true')
    parser.add_argument('-list_videos', action='store_true')
    parser.add_argument('-list_keypoints', action='store_true')
    args = parser.parse_args()

    if args.list_projects:
        list_table('projects')
    if args.list_videos:
        list_table('videos')
    if args.list_keypoints:
        list_table('keypoint_positions')