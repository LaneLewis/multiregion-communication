import numpy as np
class EulerVariables():
    def __init__(self,timeStart,timeEnd,divisions):
        self.timeStart = timeStart
        self.timeEnd = timeEnd
        self.divisions = divisions
        self.time = np.linspace(timeStart,timeEnd,divisions)
        self.timeDelta = (timeEnd - timeStart)/divisions
    