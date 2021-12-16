# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys

from numpy.lib.histograms import _histogram_bin_edges_dispatcher
from .image_viewer import ImageViewer
import cv2 as cv 

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, image_shape, update_ms, config):
        self.image_shape = image_shape
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        #image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, tuple(image_shape), "Track Visualization")
        self.viewer.thickness = 2
        self.frame_idx = 0
        self.config = config

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        self.frame_idx += 1
        frame_callback(self, self.frame_idx)
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.frame_idx += 1
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            #if not track.is_confirmed() or track.time_since_update > 0:
            #    continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            ## draw current rectangle
            try:
                _active = int(track.active)
                _score = str(track.score)[:4]
            except:
                _active = int(track.is_confirmed())
                _score = ""
            label = 'id:%sM:%iA:%iS:%s' % (str(track.track_id), track.time_since_update, _active, _score)
            self.viewer.rectangle(
                *np.array(track.to_tlwh()).astype(np.int), label=label)
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)

            ## draw lines of history 
            if hasattr(track, 'last_means'):
                if len(track.last_means)>2:
                    _stop = -1 
                    if 'track_tail' in self.config and self.config['track_tail'] > 0:
                        _stop = max(1,len(track.last_means) - self.config['track_tail'])
                    for i in range(len(track.last_means)-1, _stop, -1):
                        px, py = track.last_means[i-1][:2]
                        x,   y = track.last_means[i  ][:2]
                        #if np.sqrt( (px-x)**2 + (py-y)**2 ) < 50:
                        px,py,x,y = [int(round(c)) for c in [px,py,x,y]] 
                        self.viewer.image = cv.line(self.viewer.image,(px,py),(x,y),self.viewer.color,thickness=2)
            elif hasattr(track, 'history'):
                if len(track.history)>2:
                    _stop = -1 
                    if 'track_tail' in self.config and self.config['track_tail'] > 0:
                        _stop = max(1,len(track.history) - self.config['track_tail'])
                    for i in range(len(track.history)-1, _stop, -1):
                        p = track.history[i-1]['bbox']
                        q = track.history[i  ]['bbox']
                        px,py = p[0]+p[2]//2, p[1]+p[3]//2
                        x, y = q[0]+q[2]//2, q[1]+q[3]//2
                        #if np.sqrt( (px-x)**2 + (py-y)**2 ) < 50:
                        px,py,x,y = [int(round(c)) for c in [px,py,x,y]] 
                        self.viewer.image = cv.line(self.viewer.image,(px,py),(x,y),self.viewer.color,thickness=2)