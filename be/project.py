from multitracker.be import dbconnection
from multitracker.be import util 

"""
    === Project Handler ===
    create project logic with cli 
    
    python3.7 -m multitracker.be.project -name A -manager ADolokov -keypoint_names nose,tail
"""

def create_project(name, manager, keypoint_names):
    conn = dbconnection.DatabaseConnection()
    query = """
        insert into projects (name, manager, keypoint_names, created_at) values(?,?,?,?);
    """
    print('keypoint_names_str',keypoint_names)
    
    keypoint_names_str = conn.list_sep.join(keypoint_names)
    
    values = (name,manager, keypoint_names_str, util.get_now())
    conn.cur.execute(query, values)
    conn.commit()
    project_id = conn.get_last_id()
    print('[*] created project %i: name: %s, manager: %s, keypoint_names: [%s]' % (project_id,name,manager,', '.join(keypoint_names_str.split(conn.list_sep))))

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-name',type=str,required=True)
    parser.add_argument('-manager',type=str,required=True)
    parser.add_argument('-keypoint_names',type=str,required=True)
    args = parser.parse_args()
    
    create_project(args.name,args.manager,args.keypoint_names.split(','))
