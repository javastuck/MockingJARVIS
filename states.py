import pickle
import pyttsx
import cv2
import time
import freenect
from motion_detector import MotionDetector 
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
#from Mixins.CameraIter import Camera

class State(object):
    def identify(self):
        print "Currently helping ", self.attending
    def proceed(self):
        pass
    def revert(self):
        pass
'''		
    def action(self)
        pass
'''
        
        
#Refactor State class to force implementation of an action() method
# in order to make state transitions and actions more object oriented/pythonic


class DetectingMotion(State):

    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Detecting Motion..."
    def detect_motion(self):
        print self.activity
        self.jarvis.motion_detector.detect_motion()
        print "Detected Motion"
        self.proceed()
            
    def proceed(self):
        self.jarvis.state = self.jarvis.scanningstate
        self.jarvis.state.detect_faces()
    def revert(self):
        return 0 #This is the base state

        
class Scanning(State):
    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Scanning for Faces..."
    def detect_faces(self):
        faces = self.jarvis.face_detector.detect_faces()        
        print self.activity
        self.proceed(faces)
    def proceed(self,faces):
        print "Moving on"
        self.jarvis.state = self.jarvis.facestate
        self.jarvis.state.classify_face(faces)
    def revert(self):
        self.jarvis.state = self.jarvis.detectingstate

        
class FacialRecognition(State):
    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Classifying faces..."
        
    def classify_face(self,faces):
        names = self.jarvis.face_recognizer.recognize_faces(faces)
        print names
        if names:
            self.jarvis.attending = max(set(names),key=names.count)
            print self.activity
            self.proceed()
        else:
            self.revert()
    def proceed(self):
        self.jarvis.state = self.jarvis.greetingstate
        self.jarvis.state.greet_roommate()
    def revert(self):
        self.jarvis.state = self.jarvis.detectingstate
        self.jarvis.state.detect_motion()
        
class GreetRoommate(State):
    def __init__(self,jarvis):
        
        self.jarvis = jarvis
        self.activity = "Greeting the roommate"
        
        #Initialize text to speech bit
        self.voice = pyttsx.init()
        
    def greet_roommate(self):
        self.voice.setProperty('rate', 150)
        self.voice.say('Good morning, {}. The weather is... What can I do for you?'.format(self.jarvis.attending))
        # weather string r = requests.get('http://api.openweathermap.org/data/2.5/forecast?#id=4887398&APPID=f2a298c561abf394a488fda67700a579')
        # engine.say('The quick brown fox jumped over the lazy dog.')
        self.voice.runAndWait()
        self.proceed()
    def proceed(self):
        self.jarvis.state = self.jarvis.waitingstate
        print self.jarvis.state.activity
        self.jarvis.state.wait()
    def revert(self):
        self.jarvis.state = self.jarvis.facestate

        
class WaitingForTask(State):
    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Waiting for Task..."
    def wait(self):
        print "waiting"
        self.proceed()
    def proceed(self):
        self.jarvis.state = self.jarvis.servingstate
        self.jarvis.state.serve()
    def revert(self):
        self.jarvis.state = self.jarvis.detectingstate

        
class Serving(State):
    def __init__(self,jarvis):
        self.jarvis = jarvis
        self.activity = "Serving roommate"
    def serve(self):
        print self.activity
        self.proceed()
    def proceed(self):
        self.jarvis.state = self.jarvis.detectingstate
        self.jarvis.state.detect_motion()
    def revert(self):
        self.jarvis.state = self.jarvis.waitingstate



class Jarvis(object):
    #"This is Jarvis, JAy's Replacement and Virtually Intelligent Servant."
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
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
    #jarvis.attending = "Justin"
    #jarvis.state = jarvis.greetingstate
    #jarvis.state.greet_roommate()
    #This could be an integration test
    #actions = [jarvis.proceed,jarvis.revert,jarvis.proceed,jarvis.proceed]
    '''
    for action in actions:
        action()
        print jarvis.state.activity
    '''
if __name__=='__main__':
    main()

