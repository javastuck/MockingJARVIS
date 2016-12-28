from __future__ import print_function

import pickle

from time import time
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from combine_data import get_combined_data

class FaceRecognizer():
    def __init__(self,name=None):
        
        self.name = name
        
    def recognize_faces(self, faces):
    
        lfw_people = get_combined_data()
        # introspect the images arrays to find the shapes (for plotting)
        n_samples, h, w = lfw_people.images.shape
        
        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)
        X = lfw_people.data
        
        # the label to predict is the id of the person
        y = lfw_people.target
        
        ###############################################################################
        # Split into a training set and a test set using a stratified k fold
        
        # split into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        print("X_train shape:{}".format(X_train.shape))
        
        ###############################################################################
        # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
        # dataset): unsupervised feature extraction / dimensionality reduction
        n_components = 150
        
        print("Extracting the top %d eigenfaces from %d faces"
              % (n_components, X_train.shape[0]))

        t0 = time()
        pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
        print("done in %0.3fs" % (time() - t0))
        print("Testing {}\n".format(pca.transform(X_train).shape))
        print("Projecting the input data on the eigenfaces orthonormal basis")
        t0 = time()
        #print(pca.transform(X_test))
        X_test_pca = pca.transform(faces)
        
        target_names = lfw_people.target_names
        
        ###############################################################################
        # Train a SVM classification model
        
        clf = pickle.load(open('Faces.pkl', 'rb'))
        
        return clf.predict(X_test_pca)
'''
def main():
    facerec = FaceRecognizer()
    return facerec.recognize_faces(None)
a = main()
'''