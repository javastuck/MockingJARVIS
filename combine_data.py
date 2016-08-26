# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:19:06 2016

@author: jstuck
"""

import os
import numpy as np
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=1)

registered_names = {'Justin':7,'Conor':8}#,'Jay','Conor','Paul']
data = lfw_people.data


path_to_pics = 'C:\\Users\\jstuck\\Documents\\GitHub\\Webcam-Face-Detect'
def get_combined_data():
    for name in registered_names.keys():
        lfw_people.target_names = np.append(lfw_people.target_names,name)
        pic_folder = os.path.join(path_to_pics,name)
        for pic in os.listdir(pic_folder):
            if "Thumbs" not in pic:
                file_path = os.path.join(pic_folder, pic)   
                img = Image.open(file_path).convert('L')
                tmp = np.asarray(img).reshape(1,125,94)
                lfw_people.images = np.concatenate((lfw_people.images,tmp),axis=0)
                lfw_people.data  = np.vstack((lfw_people.data,tmp.ravel()))
                lfw_people.target = np.append(lfw_people.target,registered_names[name])
    return lfw_people

