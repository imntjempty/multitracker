import numpy as np 
import cv2 as cv 
import os 
import tensorflow as tf
import h5py

from multitracker.keypoint_detection import nets, predict
from multitracker.deepsort.deep_sort.detection import Detection
from multitracker.tracking import inference

class Detector(object):
    def __init__(self):
        pass 

    def load_model(self, h5_model):
        #self.config = config 
        self.h5_model = h5_model
        self.trained_model = tf.keras.models.load_model(h5py.File(h5_model, 'r'))
        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        self.feature_extractor = nets.EncoderPretrained({},inputs)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def detect_boundingboxes(self, vis_obj, frame_idx, sequence_dir, config):
        """
            gives tlwh and confidence
        """
        vis = vis_obj.viewer.image
        # rgb vs bgr
        R,G,B = cv.split(vis)
        merged = cv.merge((B,G,R))
        # inference

        x = tf.expand_dims(vis,axis=0)
        x = tf.cast(x,tf.float32)
        w = 1+int(2*config['img_height']/(float(x.shape[1]) / x.shape[2]))
        x = tf.image.resize(x, (2*config['img_height'],w))
        #x = x / 256.
        predicted_heatmap = self.trained_model(x,training=False)[-1]
        #predicted_heatmap = self.trained_model.predict(x)
        predicted_heatmap = predicted_heatmap.numpy()[0,:,:,:8]
        #print('x',x.numpy().min(),x.numpy().max())
        #predicted_heatmap_big = cv.resize(predicted_heatmap,vis.shape[:2][::-1])
        scale = vis.shape[1]/float(w)
        #print('predicted_heatmap',predicted_heatmap.dtype,predicted_heatmap.shape,predicted_heatmap.min(),predicted_heatmap.max())

        
        thresh = 0.3
        bboxes = [] 
        
        #for class_id, kp_name in enumerate(range(predicted_heatmap.shape[-1])):
        for class_id, kp_name in enumerate(range(1)):
            # extract each class individually
            channel = predicted_heatmap[:,:,class_id]
            candidates = inference.extract_frame_candidates(channel.copy(),thresh = thresh)
            print('[*] found %i candidates' % len(candidates),channel.min(),channel.max())
            
            for [cx,cy,proba] in candidates:
                print(cx,cy,proba)
                p = 64
                center = [cx,cy ]
                if center[0]<p:
                    center[0] = p+2
                if center[1]<p:
                    center[1] = p+2
                if center[0]>channel.shape[1]-2-p:
                    center[0] = channel.shape[1]-2-p
                if center[1]>channel.shape[0]-2-p:
                    center[1] = channel.shape[0]-2-p
                
                #feature = channel[center[1]-p:center[1]+p, center[0]-p:center[0]+p]
                #feature = cv.resize(feature, (2*p,2*p))
                #print('feat',frame_idx,channel.shape,channel.min(),channel.max(),'::',x.shape,'class',class_id,center)#,feature.shape)
                crop = x[0,center[1]-p:center[1]+p, center[0]-p:center[0]+p,:]
                crop = tf.expand_dims(crop,axis=0)
                feature = self.feature_extractor(crop)[0,:,:,:]
                
                feature = np.reshape(feature,(-1))
                #feature = np.reshape(channel[top-p:top+height+p,left-p:left+width+p],(-1))
                
                #top,left,width,height = [ab * scale for ab in [top,left,width,height]]
                q = int(16*scale)
                top = int(center[0]*scale)-q
                left = int(center[1]*scale)-q 
                width = q 
                height = q 
                detection = Detection([top,left,width,height], proba, class_id, feature)
                bboxes.append(detection)

        return bboxes 

    def detect_boundingboxes_contour(self, vis_obj, frame_idx, sequence_dir, config):
        """
            gives tlwh and confidence
        """
        vis = vis_obj.viewer.image

        # inference

        x = tf.expand_dims(vis,axis=0)
        x = tf.cast(x,tf.float32)
        w = 1+int(2*config['img_height']/(float(x.shape[1]) / x.shape[2]))
        x = tf.image.resize(x, (2*config['img_height'],w))
        predicted_heatmap = self.trained_model(x,training=False)[-1]
        predicted_heatmap = predicted_heatmap.numpy()[0,:,:,:8]
        #predicted_heatmap_big = cv.resize(predicted_heatmap,vis.shape[:2][::-1])
        scale = vis.shape[1]/w
        #print('predicted_heatmap',predicted_heatmap.dtype,predicted_heatmap.shape,predicted_heatmap.min(),predicted_heatmap.max())

        
        thresh = 0.2
        bboxes = [] 
        
        #for class_id, kp_name in enumerate(range(predicted_heatmap.shape[-1])):
        for class_id, kp_name in enumerate(range(1)):
            # extract each class individually
            channel = predicted_heatmap[:,:,class_id]
            #print(class_id,'channel',channel.min(),channel.max())
            # calculate binary mask
            mask = np.zeros_like(channel)
            mask[channel < thresh] = 0
            mask[channel >= thresh] = 1
            mask = np.uint8(mask)
            #mean = np.sum(channel * mask) / np.sum(mask/255)

            # find contours
            try:
                contours, _ = cv.findContours(mask.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            except:
                _, contours, _ = cv.findContours(mask.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # calc bbox of contours
            for cnt in contours:
                left, top, width, height = cv.boundingRect(cnt)
                if width >= 2 and height >= 2:
                    cnt_mask = mask[top:top+height,left:left+width]
                    mean = np.sum(channel[top:top+height,left:left+width] * cnt_mask) / np.sum(cnt_mask)
                    #print(len(cnt),mean,left,top,width,height)
                    
                    p = 64
                    center = [ left+width//2, top+height//2 ]
                    if center[0]<p:
                        center[0] = p+2
                    if center[1]<p:
                        center[1] = p+2
                    if center[0]>channel.shape[1]-2-p:
                        center[0] = channel.shape[1]-2-p
                    if center[1]>channel.shape[0]-2-p:
                        center[1] = channel.shape[0]-2-p
                        
                    #feature = channel[center[1]-p:center[1]+p, center[0]-p:center[0]+p]
                    #feature = cv.resize(feature, (2*p,2*p))
                    print('feat',frame_idx,channel.shape,x.shape,'class',class_id,left,center)#,feature.shape)
                    crop = x[0,center[1]-p:center[1]+p, center[0]-p:center[0]+p,:]
                    crop = tf.expand_dims(crop,axis=0)
                    feature = self.feature_extractor(crop)[0,:,:,:]
                    
                    feature = np.reshape(feature,(-1))
                    #feature = np.reshape(channel[top-p:top+height+p,left-p:left+width+p],(-1))
                    
                    top,left,width,height = [ab * scale for ab in [top,left,width,height]]
                    detection = Detection([top,left,width,height], mean, class_id, feature)
                    bboxes.append(detection)
                    #print('not too small')
                else:
                    ''#print('too small')
        output_dir = '/tmp/odetect'
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        fno = os.path.join(output_dir,'%s.png' % frame_idx)

        #print('vis',vis.shape,vis.dtype,vis.min(),vis.max(),len(bboxes))
        cv.imwrite(fno,vis)
        
        #self.bboxes = bboxes 
        return bboxes