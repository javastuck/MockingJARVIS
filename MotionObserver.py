from Observer import Observer

class MotionObserver(Observer):

	def __init__(self, name=None)
	    self.name = name
		
	def receive_message(self,sender,event,msg=None):
	    print("Received a message")