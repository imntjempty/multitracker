import os 
import numpy as np 
from random import shuffle

def make_video(frames_dir, video_file, query = "predict-%05d.png"):
    import subprocess 
    if os.path.isfile(video_file):
        os.remove(video_file)
    # great ressource https://stackoverflow.com/a/37478183
    #ffmpeg -framerate 1 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4#
    cmd = ['ffmpeg','-framerate','30']
    cmd.extend(['-pattern_type','glob','-i',"%s"%os.path.join(frames_dir,"*.png")])
    cmd.extend(['-c:v','libx264','-r','30',video_file])

    cmd = ['ffmpeg','-framerate','30','-i', os.path.join(frames_dir,query),'-vf','format=yuv420p',video_file]
    subprocess.call(cmd)


def get_colors():
    from matplotlib import colors as mcolors
    color_dicts = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)    
    colors = [] 
    for ccc in color_dicts.keys():
        #colors[ccc] = [tuple(np.int32(256*c).tolist()) for c in mcolors.to_rgba(colors[ccc])[:3]]
        color_dicts[ccc] = tuple(np.int32(256*np.array(mcolors.to_rgba(color_dicts[ccc])[:3])))
        colors.append(color_dicts[ccc])
    shuffle(colors)

    colors = [ color_dicts[k] for k in ['red','yellow','blue','green','brown','magenta','cyan','gray','purple','lightblue']]
    return colors 

def delete_video(video_id):
    from multitracker.be import dbconnection
    db = dbconnection.DatabaseConnection()
    print('''
        TODO: 
            delete all entries from db 
            delete all files from disk
    ''')



def tlbr2tlhw(tlbr):
    return [tlbr[0], tlbr[1], tlbr[2]-tlbr[0], tlbr[3]-tlbr[1]]
def tlhw2tlbr(tlhw):
    return [tlhw[0], tlhw[1], tlhw[0]+tlhw[2], tlhw[1]+tlhw[3]]
def tlhw2chw(tlhw):
    return [ tlhw[0]+tlhw[2]/2. , tlhw[1]+tlhw[3]/2., tlhw[2], tlhw[3] ]

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete_video', default=None,help='VideoID that gets deleted from database. also all files and frames are deleted.')

    '''parser.add_argument('--dir',required=True,help="Directory containing frames")
    parser.add_argument('--out',required=True,help="mp4 file to save video")
    parser.add_argument('--query',required=True,help="query for frames like predict-%05d.png")'''
    args = parser.parse_args()
    #make_video(args.dir,args.out,args.query)
    if args.delete_video is not None:
        delete_video(int(args.delete_video))