from multitracker.be import dbconnection
from multitracker.be import util 

"""
    === Project Handler ===
    create project logic with cli 
    
    python3.7 -m multitracker.be.project -name Lama1 -manager ADolokov -keypoint_names nose,body,left_ear,right_ear,left_front_feet,right_front_feet,left_back_feet,right_back_feet
"""

def create_project(name, manager, keypoint_names):
    conn = dbconnection.DatabaseConnection()
    query = """
        insert into projects (name, manager, keypoint_names, created_at) values(?,?,?,?);
    """
    print('keypoint_names_str',keypoint_names)
    
    keypoint_names_str = conn.list_sep.join(keypoint_names)
    
    values = (name,manager, keypoint_names_str, util.get_now())
    project_id = conn.insert(query, values)
    print('[*] created project %i: name: %s, manager: %s, keypoint_names: [%s]' % (project_id,name,manager,', '.join(keypoint_names_str.split(conn.list_sep))))
    return project_id 
    
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-name',type=str,required=True)
    parser.add_argument('-manager',type=str,required=True)
    parser.add_argument('-keypoint_names',type=str,required=True)
    args = parser.parse_args()
    
    create_project(args.name,args.manager,args.keypoint_names.split(','))
