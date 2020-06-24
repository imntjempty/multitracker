"""
    graph track

    http://openaccess.thecvf.com/content_cvpr_2018/papers/Girdhar_Detect-and-Track_Efficient_Pose_CVPR_2018_paper.pdf

    We represent these detections in a graph, where each detected bounding box (representing a person) in a
    frame becomes a node. We define edges to connect each box in a frame to every box in the next frame. The cost of
    each edge is defined as the negative likelihood of the two boxes linked on that edge to belong to the same person.

    We initialize tracks on the first frame and any boxes that do not get matched to an existing track instantiate a new track.

    We start from the highest confidence match, select that edge and remove the two connected nodes out of consideration. This 
    process of connecting each predicted box in the current frame with previous frame is repeatedly applied from the first to 
    the last frame of the video.

"""

def main(args):
    # load model
    tracks = [] 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--project_id',required=True,type=int)
    parser.add_argument('--video_id',required=True,type=int)
    args = parser.parse_args()
    main(args)