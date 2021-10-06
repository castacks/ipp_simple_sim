import math
import random

class SensorModel:
    def __init__(self, a, b, d, g, h):
        self.a = a
        self.b = b
        self.d = d        

    def tpr(self, range):
        return 1 / (self.a + self.b*(math.pow(math.exp, self.d*(range-self.g)))) - self.h

    def get_detection(self, range)    :
        return random.random() < self.tpr(range)