# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 14:39:14 2016

@author: jstuck
"""

import cv2
import datetime
import itertools
import time


class Camera:

    def __init__(self):
        self.stream = cv2.VideoCapture(0)
    
    def __iter__(self):
        return self
        
    def next(self):
        (ret, frame) = self.stream.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray
    
    def sample_frame(self, num_frames):
        # takes a sample set of frames of size num_frames from the video stream.
        # This is to be used for facial recognition
        return [self.next() for i in range(num_frames) if time.sleep(.5) is None]
        
            
        
camera = Camera()
timestamps = iter(datetime.datetime.now, None)
#print itertools.islice(zip(timestamps,camera),10)

for i in range(10):
    camera.next()
    time.sleep(.5)

camera.stream.release()
cv2.destroyAllWindows()

