import os 

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',required=True,help="Directory containing frames")
    parser.add_argument('--out',required=True,help="mp4 file to save video")
    parser.add_argument('--query',required=True,help="query for frames like predict-%05d.png")
    args = parser.parse_args()
    make_video(args.dir,args.out,args.query)