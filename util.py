import os 

def make_video(frames_dir, video_file):
    import subprocess 
    
    # great ressource https://stackoverflow.com/a/37478183
    #ffmpeg -framerate 1 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4#
    cmd = ['ffmpeg','-framerate','30']
    cmd.extend(['-pattern_type','glob','-i',"%s"%os.path.join(frames_dir,"*.png")])
    cmd.extend(['-c:v','libx264','-r','30',video_file])

    cmd = ['ffmpeg','-framerate','30','-i', os.path.join(frames_dir,"predict-%05d.png"),'-vf','format=yuv420p',video_file]
    subprocess.call(cmd)