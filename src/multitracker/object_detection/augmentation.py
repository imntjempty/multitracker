"""
    object detection - augmentation techniques
    hori / vertical flipping
    random cropping
    gaussian noise image
    blurring
"""

import numpy as np 
import tensorflow as tf 


def stitch_collision_free_hori(image_tensors, gt_boxes, gt_classes):
    tries, done = -1, False
    while tries < 10 and not done:
        tries += 1
        x = np.random.uniform()
        px = int(x*image_tensors.shape[2])
        found_collision = False 
        for i in range(len(gt_boxes)):
            for j in range(len(gt_boxes[i])):
                if gt_boxes[i][j][1] < x and gt_boxes[i][j][3] > x: # collision!
                    found_collision = True
        if not found_collision:
            done = True 
    if done:
        #print('stitch_collision_free_hori',tries,x,px)
        ins = image_tensors.shape
        ## shuffle first B/2 with last B/2
        B = image_tensors.shape[0]
        a = image_tensors[:B//2,:,:,:]
        b = image_tensors[B//2:,:,:,:]
        a1 = a[:,:,:px,:]
        a2 = a[:,:,px:,:]
        b1 = b[:,:,:px,:]
        b2 = b[:,:,px:,:]
        a = tf.concat((a1,b2),axis=2)
        b = tf.concat((b1,a2),axis=2)
        image_tensors = tf.concat((a,b),axis=0)
        new_boxes = [ [] for _ in range(len(gt_boxes))]
        new_classes = [ [] for _ in range(len(gt_classes))]
        for i in range(B//2):
            for j in range(len(gt_boxes[i])):
                if gt_boxes[i][j][3] < x: # is on left side
                    new_boxes[i].append( gt_boxes[i][j] )
                    new_classes[i].append( gt_classes[i][j])
                else:
                    new_boxes[i+B//2].append( gt_boxes[i][j] )
                    new_classes[i+B//2].append( gt_classes[i][j])
        for i in range(B//2,B,1):
            for j in range(len(gt_boxes[i])):
                if gt_boxes[i][j][3] < x: # is on left side
                    new_boxes[i].append( gt_boxes[i][j] )
                    new_classes[i].append( gt_classes[i][j] )
                else:
                    new_boxes[i-B//2].append( gt_boxes[i][j] )
                    new_classes[i-B//2].append( gt_classes[i][j] )
        for i in range(B):
            new_boxes[i] = tf.convert_to_tensor(new_boxes[i])
        return image_tensors, new_boxes, new_classes
    else:
        return image_tensors, gt_boxes, gt_classes


def stitch_collision_free_verti(image_tensors, gt_boxes, gt_classes):
    tries, done = -1, False
    while tries < 10 and not done:
        tries += 1
        y = np.random.uniform()
        py = int(y*image_tensors.shape[1])
        found_collision = False 
        for i in range(len(gt_boxes)):
            for j in range(len(gt_boxes[i])):
                if gt_boxes[i][j][0] < y and gt_boxes[i][j][2] > y: # collision!
                    found_collision = True
        if not found_collision:
            done = True 
    if done:
        #print('stitch_collision_free_verti',tries,y,py)
        ins = image_tensors.shape
        ## shuffle first B/2 with last B/2
        B = image_tensors.shape[0]
        a = image_tensors[:B//2,:,:,:]
        b = image_tensors[B//2:,:,:,:]
        a1 = a[:,:py,:,:]
        a2 = a[:,py:,:,:]
        b1 = b[:,:py,:,:]
        b2 = b[:,py:,:,:]
        a = tf.concat((a1,b2),axis=1)
        b = tf.concat((b1,a2),axis=1)
        image_tensors = tf.concat((a,b),axis=0)
        new_boxes = [ [] for _ in range(len(gt_boxes))]
        new_classes = [ [] for _ in range(len(gt_classes))]
        for i in range(B//2):
            for j in range(len(gt_boxes[i])):
                if gt_boxes[i][j][2] < y: # is on left side
                    new_boxes[i].append( gt_boxes[i][j] )
                    new_classes[i].append( gt_classes[i][j])
                else:
                    new_boxes[i+B//2].append( gt_boxes[i][j] )
                    new_classes[i+B//2].append( gt_classes[i][j])
        for i in range(B//2,B,1):
            for j in range(len(gt_boxes[i])):
                if gt_boxes[i][j][2] < y: # is on left side
                    new_boxes[i].append( gt_boxes[i][j] )
                    new_classes[i].append( gt_classes[i][j] )
                else:
                    new_boxes[i-B//2].append( gt_boxes[i][j] )
                    new_classes[i-B//2].append( gt_classes[i][j] )
        for i in range(B):
            new_boxes[i] = tf.convert_to_tensor(new_boxes[i])
        return image_tensors, new_boxes, new_classes
    else:
        return image_tensors, gt_boxes, gt_classes

def vflip(image_tensors, gt_boxes):
    """ vertical flipping """
    image_tensors = image_tensors[:,::-1,:,:]
    for ii in range(len(gt_boxes)):
        gt_boxes[ii] = tf.stack([1.-gt_boxes[ii][:,2],gt_boxes[ii][:,1],1.-gt_boxes[ii][:,0],gt_boxes[ii][:,3]],axis=1)
    return image_tensors, gt_boxes

def hflip(image_tensors, gt_boxes):
    """ horizontal flipping """
    image_tensors = image_tensors[:,:,::-1,:]
    for ii in range(len(gt_boxes)):
        gt_boxes[ii] = tf.stack([gt_boxes[ii][:,0],1.-gt_boxes[ii][:,3],gt_boxes[ii][:,2],1.-gt_boxes[ii][:,1]],axis=1)
    return image_tensors, gt_boxes

def gaussian_noise(image_tensors, gt_boxes):
    """ gaussian noise """
    image_tensors = tf.keras.layers.GaussianNoise(np.random.uniform(25))(image_tensors)
    return image_tensors, gt_boxes

def rot90(image_tensors, gt_boxes): # counter clockwise
    image_tensors = tf.image.rot90(image_tensors) 
    for ii in range(len(gt_boxes)): # y1,x1,y2,x2
        gt_boxes[ii] = tf.stack([1.-gt_boxes[ii][:,3],gt_boxes[ii][:,0],1.-gt_boxes[ii][:,1],gt_boxes[ii][:,2]],axis=1)
    return image_tensors, gt_boxes

def random_rot90(image_tensors, gt_boxes):
    """ randomly rotate 90, 180 or 270 deg """
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
    #if np.random.uniform() > 0.5:
    #    image_tensors = tf.image.random_jpeg_quality(image_tensors,30,100)
            
    return image_tensors, gt_boxes

def mixup(image_tensors, gt_boxes, gt_classes):
    alpha = np.random.uniform(0.33,0.66)
    B = image_tensors.shape[0]
    image_tensors = image_tensors * alpha + image_tensors[::-1,:,:,:] * (1. - alpha)
    for i in range(B//2):
        gt_boxes[i] = tf.concat((gt_boxes[i], gt_boxes[B-i-1]),0)
        gt_classes[i] = tf.concat((gt_classes[i], gt_classes[B-i-1]),0)
        gt_boxes[B-i-1] = tf.concat((gt_boxes[i], gt_boxes[B-i-1]),0)
        gt_classes[B-i-1] = tf.concat((gt_classes[i], gt_classes[B-i-1]),0)
    return image_tensors, gt_boxes, gt_classes

def augment(config, image_tensors, gt_boxes, gt_classes):
    bhflip, bvflip, bmixup, brot90 = False,False,False,False 
    if config['object_augm_flip']:
        if np.random.uniform() > 0.5:
            image_tensors, gt_boxes = hflip(image_tensors, gt_boxes)
            bhflip = True 
        if np.random.uniform() > 0.5:
            image_tensors, gt_boxes = vflip(image_tensors, gt_boxes)
            bvflip = True 

    if len(gt_boxes)>0 and config['object_augm_stitch'] and np.random.uniform() < 0.2:
        if np.random.uniform() > .5:
            image_tensors, gt_boxes, gt_classes = stitch_collision_free_hori(image_tensors, gt_boxes, gt_classes)
        else:
            image_tensors, gt_boxes, gt_classes = stitch_collision_free_verti(image_tensors, gt_boxes, gt_classes)

    if config['object_augm_mixup'] and np.random.uniform() < 0.5:
        image_tensors, gt_boxes, gt_classes = mixup(image_tensors, gt_boxes, gt_classes)
        bmixup = True 

    if config['object_augm_gaussian'] and np.random.uniform() > 0.5:
        image_tensors, gt_boxes = gaussian_noise(image_tensors, gt_boxes)
    if config['object_augm_rot90'] and np.random.uniform() > 0.5:
        image_tensors, gt_boxes = random_rot90(image_tensors, gt_boxes)
        brot90 = True 
    if config['object_augm_crop'] and np.random.uniform() > 0.5:
        image_tensors, gt_boxes = random_crop(image_tensors, gt_boxes)
    if config['object_augm_image'] and np.random.uniform() > 0.5:
        image_tensors, gt_boxes = random_image_transformation(image_tensors, gt_boxes)
    
    #print('[*] augment','hflip',bhflip,'vflip',bvflip,'rot90',brot90,'mixup',bmixup)
    return image_tensors, gt_boxes, gt_classes

def test():
    pass 
if __name__ == "__main__":
    test()