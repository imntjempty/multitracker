"""
    here we draw heatmaps for keypoints

    we use this for inputs as conv nets or draw nice visualizations with it
"""

import numpy as np 
import cv2 as cv 
#import tensorflow as tf 
import os 
from random import shuffle 
from multitracker.be import dbconnection
from glob import glob 

def gaussian_k(x0,y0,sigma, height, width):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float) ## (width,)
    y = np.arange(0, height, 1, float)[:, np.newaxis] ## (height,1)
    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

def generate_hm(height, width ,landmarks, keypoint_names, s=None):
    ## https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html 
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing landmarks
        maxlenght : Lenght of the Bounding Box
    """
    if s is None:
        s = height / 100.
    hm = 0.* np.ones((height, width, len(keypoint_names)), dtype = np.float32)
    for i in range(len(landmarks)):
        idx = keypoint_names.index(landmarks[i][2])
        x = gaussian_k(landmarks[i][0],
                                landmarks[i][1],
                                s,height, width)
        hm[:,:,idx] += x
        #hm[:,:,idx][x>0.1] = x 
    #hm[hm<0.3] = 128.
    hm[hm>1.0] = 1.0
    return hm


def vis_heatmap(image, keypoint_names, keypoints, horistack=True, apply_contrast_stretching=False, apply_histogram_equalization=False, apply_adaptive_equalization=False ):
    hm = generate_hm(image.shape[0], image.shape[1] , [ [int(kp[2]),int(kp[3]),kp[0]] for kp in keypoints ], keypoint_names)
    
    im = image 
    from skimage import exposure
    if apply_contrast_stretching:
        # Contrast stretching
        p2, p98 = np.percentile(im, (2, 98))
        im = exposure.rescale_intensity(im, in_range=(p2, p98))

    if apply_histogram_equalization: 
        # Histogram Equalization
        im = exposure.equalize_hist(im)
        im = np.uint8(np.around(255 * im))

    if apply_adaptive_equalization:
        # Adaptive Equalization
        im = exposure.equalize_adapthist(im, clip_limit=0.03)
        im = np.uint8(np.around(255 * im))

    if not horistack:
        # overlay
        #hm = np.uint8(255. * hm[:,:,:3])
        #vis = np.uint8( hm//2 + image//2 )     
        vis = np.uint8(np.dstack((im,np.uint8(255. * hm))))
    else:
        # make horizontal mosaic - image and stacks of 3
        n = 3 * (hm.shape[2]//3) + 3
        while hm.shape[2] < n:
            hm = np.dstack((hm,np.zeros(im.shape[:2])))
        vis = im 
        hm = np.uint8(255. * hm)
        for i in range(0,n,3):
            vis = np.hstack((vis, np.dstack((hm[:,:,i],hm[:,:,i+1],hm[:,:,i+2] ) )))

    return vis 

def style_augment(directory, style_directory = os.path.expanduser("~/data/multitracker/styleimages")):
    import subprocess
    if not os.path.isdir(style_directory) or len(glob(os.path.join(style_directory,'*'))) == 0:
        # download art images
        urls = [
            'https://cdn.kunstschaetzen.de/wp-content/uploads/2019/06/popart-kunst.jpg',
            'https://i.pinimg.com/originals/fe/41/5f/fe415f65c3641c1a67b000aa7a4ddb36.jpg',
            'https://news.artnet.com/app/news-upload/2019/12/5db820a075ba3.jpg',
            'https://images.unsplash.com/photo-1543857778-c4a1a3e0b2eb?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80',
            'https://lh6.ggpht.com/HlgucZ0ylJAfZgusynnUwxNIgIp5htNhShF559x3dRXiuy_UdP3UQVLYW6c=s1200',
            'https://www.ourartworld.com/art/wp-content/uploads/2018/07/94e5c8f8a38382c6750f26a2467ad670--bright-art-lonely-heart.jpg',
            'https://images.squarespace-cdn.com/content/v1/57762ed29f7456e1b6e8febd/1580403012581-RD72EGBP08TA7V50Z5WU/ke17ZwdGBToddI8pDm48kFJo1x-0SDSXZIbvHdtV7zB7gQa3H78H3Y0txjaiv_0fDoOvxcdMmMKkDsyUqMSsMWxHk725yiiHCCLfrh8O1z4YTzHvnKhyp6Da-NYroOW3ZGjoBKy3azqku80C789l0p5uBJOnOmCWBN4JfYsIDyTk732yQxGFdJ5_SwaFk5-9rqZMSEJs3dyKz5eVslWaTA/2013+12+JIMI+Hendrix%2C+190x190cm%2C+Acrylic+on+Canvas.jpg?format=500w',
            'https://www.scheublein.com/wp-content/uploads/2020/05/Slider-Blick-auf-See.jpg',
            'https://www.justcolor.net/de/wp-content/uploads/sites/5/nggallery/op-art/malbuch-fur-erwachsene-op-art-53995.jpg',
            'https://www.diebewertung.de/wp-content/uploads/2019/04/Kunst_1556194083.jpg',
            'https://cdn.mos.cms.futurecdn.net/jbCNvTM4gwr2qV8X8fW3ZB.png',
            'https://static.boredpanda.com/blog/wp-content/uploads/2017/02/IMG_20170205_220039_508-58a0129a23ada__880.jpg',
            'https://www.zdf.de/assets/teletext-dpa-image-street-art-werk-in-bristol-ein-banksy-100~2400x1350?cb=1581618822459',
            'https://www.agora-gallery.com/images/news_headers/Abstract%20art%20-%20Rebecca%20Katz.jpg',
            'https://www.staedelmuseum.de/sites/default/files/styles/col-12/public/title/startseiten_slider_podcast_dt.jpg?itok=f8xD8G5O',
            'https://onlinegallery.art/images/albums/vincent-van-gogh-the-yellow-house-(-the-street-).jpg?quality=85&type=webp&resolution=1920x0',
            'https://www.boesner.com/cache/upload/event/superClient/weber-050620-perlw0h0height.jpg',
            'https://cdn11.bigcommerce.com/s-x49po/images/stencil/1280x1280/products/31965/44808/1527844490374_Tree_by_Luis_David-mixed_on_masonite-23.5___X_31.5__-1550_usd.-__81859.1528180230.jpg?c=2&imbypass=on'
        ]
        for i, url in enumerate(urls):
            fn = os.path.join(style_directory,'%i.jpg' % i)
            subprocess.call(['wget',url,'-O',fn])
        print('[*] downloaded %i style images' % len(urls))
    
    style_images = glob(os.path.join(style_directory,'*'))
    
    files = glob(os.path.join(directory,'*.png'))
    for i, f in enumerate(files):
        for j, fs in enumerate(style_images):
            fo = f.replace('.png','style%i_%i.png' % (i,j))
            # python main.py -content_path 'content_example.jpg' -style_path 'style_example.jpg'
            subprocess.call(['python3.7',os.path.expanduser('~/github/tensorflow-2-style-transfer/main.py'),'-content_path',"%s" % f, '-style_path',"%s" % fs,'-output_dir',"%s" % directory])
        

def randomly_drop_visualiztions(project_id, dst_dir = '/tmp/keypoint_heatmap_vis', num = -1, horistack=True ):
    # take random frames from the db and show their labeling as gaussian heatmaps
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    db = dbconnection.DatabaseConnection()
    
    keypoint_names = db.get_keypoint_names(project_id)
    frame_idxs = []
    while len(frame_idxs) == 0:
        video_id = db.get_random_project_video(project_id)
        if video_id is None:
            raise Exception("[ERROR] no video found for project!")

        # first get all frames 
        q = "select frame_idx from keypoint_positions where video_id=%i;" % video_id
        db.execute(q)
        frame_idxs = [x[0] for x in db.cur.fetchall()]
        frame_idxs = list(set(frame_idxs))
    shuffle(frame_idxs)

    if num > 0:
        frame_idxs = frame_idxs[:min(num,len(frame_idxs))]

    for mode in ['train','test']:
            mode_dir = os.path.join(dst_dir,mode)
            if not os.path.isdir(mode_dir):
                os.makedirs(mode_dir)

    for i in range(len(frame_idxs)):
        filepath = os.path.expanduser("~/data/multitracker/projects/%i/%i/frames/train/%s.png" % (int(project_id), int(video_id), frame_idxs[i]))
        q = "select keypoint_name, individual_id, keypoint_x, keypoint_y from keypoint_positions where video_id=%i and frame_idx='%s' order by individual_id, keypoint_name desc;" % (video_id,frame_idxs[i])
        db.execute(q)
        keypoints = [x for x in db.cur.fetchall()]
        #print('frame',frame_idxs[i],'isfile',os.path.isfile(filepath),filepath)
        #for kp in keypoints:
        #    print(kp)
        mode = 'train' if np.random.uniform() > 0.2 else 'test'
        if os.path.isfile(filepath):
            im = cv.imread(filepath)
            vis = vis_heatmap(im, keypoint_names, keypoints, horistack = horistack)
            vis_path = os.path.join(dst_dir,mode,'%s.png' % frame_idxs[i] )
            cv.imwrite(vis_path, vis)

            if mode == 'train': # only train
                vis = vis_heatmap(im, keypoint_names, keypoints, horistack = horistack, apply_contrast_stretching=True)
                vis_path = os.path.join(dst_dir,mode,'%sc0.png' % frame_idxs[i] )
                cv.imwrite(vis_path, vis)

                vis = vis_heatmap(im, keypoint_names, keypoints, horistack = horistack, apply_histogram_equalization=True)
                vis_path = os.path.join(dst_dir,mode,'%sc1.png' % frame_idxs[i] )
                cv.imwrite(vis_path, vis)
                
                vis = vis_heatmap(im, keypoint_names, keypoints, horistack = horistack, apply_adaptive_equalization=True)
                vis_path = os.path.join(dst_dir,mode,'%sc2.png' % frame_idxs[i] )
                cv.imwrite(vis_path, vis)
        

if __name__ == '__main__':
    project_id = 1
    randomly_drop_visualiztions(project_id, horistack = False)