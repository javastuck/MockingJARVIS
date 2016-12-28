# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 18:26:39 2016

@author: jstuck
"""

#This is a mixin to register a new person for Jarvis' facial recognition

import cv2
import os
import numpy as np
from PIL import Image

path_to_pics = 'C:\\Users\\jstuck\\Documents\\GitHub\\Webcam-Face-Detect'

class Registrar():
    def register_face(self, faces, name):
        registered_users = {}
        with open(os.path.join(path_to_pics,'face_labels.txt')) as face_labels:
            for line in face_labels:
                name, label = line.split(',')
                registered_users[name] = int(label)
        print registered_users
        
def main():
    a = Registrar()
    a.register_face(1,2)
main()        