from __future__ import print_function

from time import time
from classifier import infer
classifier_model='/home/jstuck/Desktop/Jarvis/MockingJARVIS/Faces/feature_dir/classifier.pkl'

imgDim=96


class FaceRecognizer():
    def __init__(self,name=None):
        
        self.name = name
        
    def recognize_faces(self, faces):
        results = infer(classifier_model,faces)
        return results
