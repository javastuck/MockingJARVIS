from abc import ABCMeta, abstractmethod
 
class Observer(object):
    __metaclass__ = ABCMeta
 
    @abstractmethod
    def receive_message(self, sender,event,message):
        pass