"""
    recognition video prediction
    load a trained model and predict and visualize each frame

    example call
        python3.7 -m multitracker.keypoint_detection.predict -model /home/alex/checkpoints/keypoint_tracking/2020-05-28_07-51-11 -project 2
"""

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 
from datetime import datetime
import cv2 as cv 
import h5py

from multitracker.keypoint_detection import heatmap_drawing, model 

# <network architecture>
from tensorflow.keras.applications.resnet_v2 import preprocess_input

def get_random_colors(n):
    colors = []
    for i in range(n):
        r = int(np.random.uniform(0,255))
        g = int(np.random.uniform(0,255))
        b = int(np.random.uniform(0,255))
        colors.append((r,g,b))
    colors = np.int32(colors)
    return colors 
colors = get_random_colors(100)

def get_project_frames(config, project_id = None):
    if project_id is None:
        project_id = config['project_id']
    
    def load_im(image_file):
        print(image_file)
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image,channels=3)
        image = tf.cast(image,tf.float32)
        return image 
    
    frames_dir = get_project_frame_test_dir(project_id)
    frames = tf.data.Dataset.list_files(os.path.join(frames_dir,'*.png'),shuffle=False).map(load_im, num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(config['batch_size'])#.prefetch(4*config['batch_size'])#.cache()
    return frames 

def get_project_frame_test_dir(project_id):
    return os.path.expanduser('~/data/multitracker/projects/%i/%i/frames/test' % (project_id,project_id))

def predict(config, checkpoint_path, project_id):
    project_id = int(project_id)
    output_dir = '/tmp/multitracker/predictions/%i/%s' % (project_id, checkpoint_path.split('/')[-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
 
    path_model = os.path.join(checkpoint_path,'trained_model.h5')
    trained_model = tf.keras.models.load_model(h5py.File(path_model, 'r'))

    frames_dir = get_project_frame_test_dir(project_id)
    config['input_image_shape'] = cv.imread(glob(os.path.join(frames_dir,'*.png'))[0]).shape[:2]

    config['n_inferences'] = 5
    
    frames = get_project_frames(config, project_id)
    #frames = model.load_raw_dataset(config,mode='custom',image_directory = )
    cnt_output = 0 
    for x in frames:
        xsmall = x 
        w = config['img_height']*x.shape[2]//x.shape[1]
        xsmall = tf.image.resize(x,(config['img_height'],w))
        xsmall = xsmall[:,:,(xsmall.shape[2]-xsmall.shape[1])//2:xsmall.shape[2]//2+xsmall.shape[1]//2,:] # center square crop
        
        if config['n_inferences'] == 1:
            y = trained_model.predict(xsmall)
        else:
            y = trained_model(xsmall, training=True) / config['n_inferences']
            for ii in range(config['n_inferences']-1):
                y += trained_model(xsmall, training=True) / config['n_inferences']
            
        #xn, yn = x.numpy(),y 
        #print(x.shape,y.shape,xsmall.shape, xn.min(),xn.max(),yn.min(),yn.max())

        for b in range(x.shape[0]): # iterate through batch of frames
            #vis_frame = np.zeros([x.shape[1],x.shape[2],3])
            vis_frame = np.zeros([config['img_height'],config['img_width'],3])
            for c in range(y.shape[3]): # iterate through each channel for this frame
                feature_map = y[b,:,:,c]
                feature_map = np.expand_dims(feature_map, axis=2)
                feature_map = colors[c] * feature_map 
                vis_frame += feature_map 
            vis_frame = np.around(vis_frame)
            vis_frame = np.uint8(vis_frame)
            #print('1vis_frame',vis_frame.shape,vis_frame.dtype,vis_frame.min(),vis_frame.max())
            overlay = np.uint8(xsmall[b,:,:,:]//2 + vis_frame//2)
            vis_frame = np.hstack((overlay,vis_frame))
            fp = os.path.join(output_dir,'predict-{:05d}.png'.format(cnt_output))
            cv.imwrite(fp, vis_frame)
            
            cnt_output += 1 

def predictold(config, checkpoint_path, project_id):
    project_id = int(project_id)
    output_dir = '/tmp/multitracker/predictions/%i/%s' % (project_id, checkpoint_path.split('/')[-1])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    #dataset_test = model.load_raw_dataset(config,'test')
    
    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 3])
    inputsprep = inputs# preprocess_input(inputs)
    encoder = model.Encoder(config,inputsprep)
    #print('[*] hidden representation',encoder.outputs[0].get_shape().as_list())
    heatmaps = model.Decoder(config,encoder)

    trained_model = tf.keras.models.Model(inputs = encoder.input, outputs = heatmaps) #dataset['train'][0],outputsdataset['train'][1])
    #trained_model.summary() 

    ckpt = tf.train.Checkpoint(trained_model = trained_model, optimizer = tf.keras.optimizers.Adam(1e-5))
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    assert ckpt_manager.latest_checkpoint, ("ERROR! could not find checkpoint in "+checkpoint_path)
    status = ckpt.restore(ckpt_manager.latest_checkpoint)#.expect_partial()
    status.assert_consumed()
    print('[*] checkpoint restored',ckpt_manager.latest_checkpoint)

    # load frame datas
    #frames = get_project_frames(config, project_id=project_id)
    frames = model.load_raw_dataset(config,mode='test')
    cnt_output = 0 
    for x in frames:
        xs = x.shape 
        #w = config['img_height']*xs[2]//xs[1]
        #print(xs,'w',w)
        #xsmall = tf.image.resize(x,(config['img_height'],w))
        #xsmall = xsmall[:,:,(xsmall.shape[2]-xsmall.shape[1])//2:-(xsmall.shape[2]-xsmall.shape[1])//2,:] # center crop
        
        
        
        xsmallori = crop 
        xsmall = preprocess_input(xsmallori)
        y = trained_model(xsmall, training=False)
        #y = tf.nn.softmax(y)
        
        xn , yn = x.numpy(), y.numpy()
        print(x.shape,y.shape,xsmall.shape, xn.min(),xn.max(),yn.min(),yn.max())
        s = y.shape


        for b in range(s[0]): # iterate through batch of frames
            vis_frame = np.zeros([s[1],s[2],3])
            for c in range(s[3]): # iterate through each channel for this frame
                h = y[b,:,:,c]
                #h = tf.cast(h,'uint8')
                hh = np.zeros(h.shape,'uint8')
                hh[h>0.5]=1
                #h = hh 
                
                h = np.expand_dims(h, axis=2)
                h = colors[c] * h 
                
                #print('h',b,c,h.shape)
                #pad = np.zeros((h.shape[0],(y.shape[2]-y.shape[1])//2,3),'uint8')
                #print('h',h.shape,'vis',vis_frame.shape,'pad',pad.shape)
                #vis_frame += np.concatenate((pad,h,pad),axis=1)
                vis_frame += h 
            #vis_frame[vis_frame>255.] = 255. 
            print('vis',vis_frame.shape, vis_frame.min(),vis_frame.max())
            vis_frame = np.uint8(vis_frame)

            # overlay with input image 
            #vis_frame = np.uint8(vis_frame // 2 + xn[b,:,:,:] // 2)
            vis_frame = np.hstack((xsmallori[b,:,:,:],vis_frame))
            f = os.path.join(output_dir,'predict-{:05d}.png'.format(cnt_output))
            cv.imwrite(f, vis_frame)
            print(cnt_output, f)
            cnt_output += 1 



def main(checkpoint_path, project_id):
    config = model.get_config()
    predict(config, checkpoint_path, project_id)


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-model')
    parser.add_argument('-project')
    args = parser.parse_args()
    main(args.model, args.project)