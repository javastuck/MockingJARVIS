import pickle
import pyttsx
import cv2
import time
from motion_detector import MotionDetector 
from face_detector import FaceDetector
class State(object):
    def identify(self):
        print "Currently helping ", self.attending
    def proceed(self):
        pass
'''		
    def action(self)
        pass
'''
    def revert(self):
        pass
        
        
#Refactor State class to force implementation of an action() method
# in order to make state transitions and actions more object oriented/pythonic


class DetectingMotion(State):

    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Detecting Motion..."
    def detect_motion(self):
        print self.activity
        self.jarvis.motion_detector.detect_motion(self.jarvis.camera)
        print "Detected Motion"
        self.proceed()
            
    def proceed(self):
        self.jarvis.state=self.jarvis.scanningstate
        self.jarvis.state.detect_faces()
    def revert(self):
        return 0 #This is the base state

        
class Scanning(State):
    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Scanning for Faces..."
    def detect_faces():
        print self.activity
        faces = 
        
        
    def proceed(self):
        self.jarvis.state =  self.jarvis.facestate
        self.jarvis.state.classify_face(faces)
    def revert(self):
        self.jarvis.state = self.jarvis.detectingstate

        
class FacialRecognition(State):
    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Classifying faces..."
        self.classifier = pickle.load(open('Faces.pkl', 'rb'))
        
    def classify_face(faces):
        names = self.classifier.predict(faces)
        
        print self.activty
        
    def proceed(self):
        self.jarvis.state = self.jarvis.greetingstate
    def revert(self):
        self.jarvis.state = self.jarvis.detectingstate

        
class GreetRoommate(State):
    def __init__(self,jarvis):
        
        self.jarvis = jarvis
        self.activity = "Greeting the roommate"
        
        #Initialize text to speech bit
        self.voice = pyttsx.init()
        
    def greet_roommate(self):
        self.voice.setProperty('rate', 90)
        self.voice.say('Greetings, {}'.format(self.jarvis.attending))
        #engine.say('The quick brown fox jumped over the lazy dog.')
        self.voice.runAndWait()
        self.proceed()
    def proceed(self):
        self.jarvis.state = self.jarvis.waitingstate
    def revert(self):
        self.jarvis.state = self.jarvis.facestate

        
class WaitingForTask(State):
    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Waiting for Task..."
    
    def proceed(self):
        self.jarvis.state = self.jarvis.servingstate
    def revert(self):
        self.jarvis.state = self.jarvis.detectingstate

        
class Serving(State):
    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Serving roommate"
    def proceed(self):
        self.jarvis.state = self.jarvis.waitingstate
    def revert(self):
        self.jarvis.state = self.jarvis.waitingstate



class Jarvis(object):
    #"This is Jarvis, JAy's Replacement and Virtually Intelligent Servant."
    def __init__(self):
        #observer = Observable()
        #motion_detector = MotionObserver('Motion Detector')
        #observer.register(motion_detector)
        self.camera = cv2.VideoCapture(0)
        time.sleep(0.5)
        self.motion_detector = MotionDetector()
        self.face_detector = FaceDetector()
        self.detectingstate = DetectingMotion(self)
        self.scanningstate = Scanning(self)
        
        self.facestate = FacialRecognition(self)
        self.greetingstate = GreetRoommate(self)
        self.waitingstate = WaitingForTask(self)
        self.servingstate = Serving(self)
        self.state = self.detectingstate

        
    def proceed(self):
        self.state.proceed()

    def revert(self):
        self.state.revert()



def main():
    jarvis = Jarvis()
    jarvis.state.detect_motion()
    jarvis.attending = "Justin"
    jarvis.state = jarvis.greetingstate
    jarvis.state.greet_roommate()
    #This could be an integration test
    #actions = [jarvis.proceed,jarvis.revert,jarvis.proceed,jarvis.proceed]
    '''
    for action in actions:
        action()
        print jarvis.state.activity
    '''
if __name__=='__main__':
    main()

