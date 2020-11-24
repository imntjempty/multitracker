"""
    object detection - augmentation techniques
    hori / vertical flipping
    random cropping
    gaussian noise image
    blurring
"""

import numpy as np 
import tensorflow as tf 

def vflip(image_tensors, gt_boxes):
    image_tensors = image_tensors[:,::-1,:,:]
    for ii in range(len(gt_boxes)):
        gt_boxes[ii] = tf.stack([1.-gt_boxes[ii][:,2],gt_boxes[ii][:,1],1.-gt_boxes[ii][:,0],gt_boxes[ii][:,3]],axis=1)
    return image_tensors, gt_boxes

def hflip(image_tensors, gt_boxes):
    image_tensors = image_tensors[:,:,::-1,:]
    for ii in range(len(gt_boxes)):
        gt_boxes[ii] = tf.stack([gt_boxes[ii][:,0],1.-gt_boxes[ii][:,3],gt_boxes[ii][:,2],1.-gt_boxes[ii][:,1]],axis=1)
    return image_tensors, gt_boxes

def gaussian_noise(image_tensors, gt_boxes):
    image_tensors = tf.keras.layers.GaussianNoise(100)(image_tensors)
    return image_tensors, gt_boxes

def rot90(image_tensors, gt_boxes): # counter clockwise
    image_tensors = tf.image.rot90(image_tensors) 
    for ii in range(len(gt_boxes)): # y1,x1,y2,x2
        gt_boxes[ii] = tf.stack([1.-gt_boxes[ii][:,3],gt_boxes[ii][:,0],1.-gt_boxes[ii][:,1],gt_boxes[ii][:,2]],axis=1)
    return image_tensors, gt_boxes

def random_rot90(image_tensors, gt_boxes):
    for _ in range(int(np.random.uniform(4))):
        image_tensors, gt_boxes = rot90(image_tensors, gt_boxes)
    return image_tensors, gt_boxes

def random_crop(image_tensors, gt_boxes):
    H, W = image_tensors.shape[1:3]
    for ii in range(len(gt_boxes)): # y1,x1,y2,x2
        gt_boxes[ii] *= np.array([H,W,H,W])

    #Hc = int(np.random.uniform(H/2,H))
    #Wc = int(Hc*W/H)
    #ry = int(np.random.uniform(0,H-Hc))
    #rx = int(np.random.uniform(0,W-Wc))
    a = int(np.random.uniform(0.5,1.0)*min(H,W))
    y = int(np.random.uniform(0,H-a))
    x = int(np.random.uniform(0,W-a))
    
    print('TODO random_crop',H,W,Hc,Wc,'pos',rx,ry)

    cropped_boxes = [] 
    for ii in range(len(gt_boxes)): # y1,x1,y2,x2
        ''    # 

    for ii in range(len(gt_boxes)): # y1,x1,y2,x2
        gt_boxes[ii] /= np.array([Hc,Wc,Hc,Wc])
    
    return image_tensors, gt_boxes

def random_image_transformation(image_tensors, gt_boxes):
    if np.random.uniform() > 0.5:
        image_tensors = tf.image.random_saturation(image_tensors,0.5,2)
    if np.random.uniform() > 0.5:
        image_tensors = tf.image.random_brightness(image_tensors,30)
    if np.random.uniform() > 0.5:
        image_tensors = tf.image.random_contrast(image_tensors,0.5,2)
    if np.random.uniform() > 0.5:
        image_tensors = tf.image.random_hue(image_tensors,0.25)
    if np.random.uniform() > 0.5:
        image_tensors = tf.image.random_jpeg_quality(image_tensors,30,100)
            
    return image_tensors, gt_boxes

def augment(config, image_tensors, gt_boxes):
    if config['object_augm_flip']:
        if np.random.uniform() > 0.5:
            image_tensors, gt_boxes = hflip(image_tensors, gt_boxes)
        if np.random.uniform() > 0.5:
            image_tensors, gt_boxes = vflip(image_tensors, gt_boxes)
    if config['object_augm_gaussian'] and np.random.uniform() > 0.5:
        image_tensors, gt_boxes = gaussian_noise(image_tensors, gt_boxes)
    if config['object_augm_rot90'] and np.random.uniform() > 0.5:
        image_tensors, gt_boxes = random_rot90(image_tensors, gt_boxes)
    if config['object_augm_crop'] and np.random.uniform() > 0.5:
        image_tensors, gt_boxes = random_crop(image_tensors, gt_boxes)
    if config['object_augm_image'] and np.random.uniform() > 0.5:
        image_tensors, gt_boxes = random_image_transformation(image_tensors, gt_boxes)
    object_augm_image
    return image_tensors, gt_boxes

def test():
    pass 
if __name__ == "__main__":
    test()