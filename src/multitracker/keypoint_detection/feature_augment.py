"""

    augment training by using 3 random feature maps from first layer convolution

"""

import os 
import numpy as np 
import cv2 as cv 
import tensorflow as tf 
from glob import glob 

from multitracker.keypoint_detection import model 
from tensorflow.keras.applications.resnet_v2 import preprocess_input

def augment_dataset(project_id):
    config = model.get_config(project_id)
    config['input_image_shape'] = cv.imread(glob(os.path.join(config['data_dir'],'train/*.png'))[0]).shape[:2]
    h = config['input_image_shape'][0]
    w = config['input_image_shape'][1] // (2+ len(config['keypoint_names'])//3)
    print(config)
    print(h,w, 4 * w)
    
    file_list, dataset_train = model.load_raw_dataset(config,'train')
    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 3])
    inputss = preprocess_input(inputs)
    net = tf.keras.applications.ResNet152V2(input_tensor=inputss,
            include_top=False,
            weights='imagenet',
            pooling='avg')
    net.summary()
    if 0:
        for layer in net.layers:
            try:
                print(layer.name,layer.outputs[0].shape)
            except:
                pass 

    layer_name = 'conv1_conv'
    feature_activation = net.get_layer(layer_name)
    feature_extractor = tf.keras.models.Model(name="ImageNet Encoder",inputs=net.input,outputs=[feature_activation.output])
    
    output_dir = '/tmp/feature_augment'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    files = [f for f in sorted(glob(os.path.join(config['data_dir'],'train/*.png'))) if not 'augment-' in f]
    for i,f in enumerate(files):
        im = cv.imread(f)
        rgb = im[:,:w,:]
        rgb = rgb.reshape((1,) + rgb.shape)
        rgb = preprocess_input(rgb)
        features = feature_extractor(rgb,training=False).numpy()[0,:,:,:]
        features = cv.resize(features,None,None,fx=2.,fy=2.)

        print('[*] wrote %i/%i:'%(i,len(files)),im.shape,features.shape,features.min(),features.max())

        for j in range(2):
            fo = os.path.join(config['data_dir'],'train/augment-%i-%i.png' % (i,j) )
            #print(fo)
            a = np.zeros_like(features[:,:,0])
            b = np.zeros_like(features[:,:,0])
            c = np.zeros_like(features[:,:,0])
            for k in range(features.shape[-1] // 3):
                a += features[:,:,int(np.random.uniform(features.shape[-1]))]
                b += features[:,:,int(np.random.uniform(features.shape[-1]))]
                c += features[:,:,int(np.random.uniform(features.shape[-1]))]
            abc = cv.merge((a,b,c))
            thresh = 1. 
            abc[thresh > np.abs(abc)] = thresh 
            abc8 = 255. * (abc-abc.min())/(1e-5+abc.max()-abc.min())
            abc8 = np.uint8(abc8)
            abc8 = np.hstack((abc8,im[:,w:,:]))
            cv.imwrite(fo,abc8)

    '''for x,y in dataset_train:
        x = x.numpy() 
        y = y.numpy() 
        print(x.shape,x.min(),x.max(),y.shape,y.min(),y.max())'''
if __name__ == '__main__':
    test()