import os, shutil
import numpy as np
import cv2
import sys
import PIL
import time
import imutils
from PIL import Image

basewidth = 94
hsize = 125


class FaceDetector():
    def __init__(self,name=None):
        
        self.name = name
        self.faces = []
        self.done_detecting = False
        
    def detect_faces(self, camera):
    
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        counter=0
        faceInFrame = False
        
        #refactor to make loop break more elegant
        
        while True:
            # Capture frame-by-frame
            counter += 1
            (ret, frame) = camera.read()
            
            #frame = imutils.resize(frame, width=94, height=125)    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            '''
            if len(faces)!=0:
                if faceInFrame==False:        
                    engine.say("Welcome home, fuck face")
                    engine.runAndWait()
                    faceInFrame = True      
            else:
                faceInFrame=False
            '''
            #refactor this to remove the breaks
            if len(self.faces)<11:
                
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow('Video', frame)                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, (94,125))
                    #cv2.imwrite(name, frame[y-int(.1*h): y + h, x: x + w])            
                    #img = frame[y-int(.1*h): y + h, x: x + w]            
                    #img = Image.open(name).convert('LA')            

                    #img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
                    #cv2.imwrite("Justin/face{}.jpeg".format(counter/5), img)
                    #frame = frame[y-int(1.1*h): y + h, x: x + w]                
                    #frame = imutils.resize(frame, width=94, height=125)  
                    #print frame.shape
                    face = np.asarray(frame).reshape(1,125,94).ravel()
                    #print face
                    self.faces.append(face)

            else:
                # When everything is done, release the capture        
                camera.release()
                cv2.destroyAllWindows()
                return np.asarray(self.faces)

            # Display the resulting frame
            #cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                camera.release()
                cv2.destroyAllWindows()
                return np.asarray(self.faces)



def main():
    camera = cv2.VideoCapture(0)
    detector = FaceDetector()
    detector.detect_faces(camera)
    faces = detector.faces
    print(faces)
    
    return faces
a = main()






