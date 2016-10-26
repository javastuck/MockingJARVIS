import os, shutil
import cv2
import sys
import PIL
import time

from PIL import Image

basewidth = 94
hsize = 125


class FaceDetector():
    def __init__(self,name=None):
        self.name = name
        self.faces = [0]
        self.done_detecting = False
    def detect_faces(self):
       
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)






        video_capture = cv2.VideoCapture(0)
        time.sleep(0.5)
        counter=0
        faceInFrame=False
        while True:
            # Capture frame-by-frame
            counter += 1
            ret, frame = video_capture.read()

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
            if len(self.faces)<11:
                
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y-int(.3*h)), (x+w, y+int(1.2*h)), (0, 255, 0), 2)
                    name ="face.png"
                    cv2.imwrite(name, frame[y-int(.1*h): y + h, x: x + w])            
                    #img = frame[y-int(.1*h): y + h, x: x + w]            
                    img = Image.open(name).convert('LA')            

                    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
                    #cv2.imwrite("Justin/face{}.jpeg".format(counter/5), img)
                    self.faces.append(img)

            else:
                break            
            # Display the resulting frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

'''        
def main():
    detector = FaceDetector()
    detector.detect_faces()

main()
'''





