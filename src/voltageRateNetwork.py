import numpy as np
from tqdm import tqdm

class VoltageRateNetwork():
    def __init__(self,weightMatrix,rateFunction,voltageDiffEq):
        self.networkSize = weightMatrix.shape[0]
        self.weightMatrix = weightMatrix
        self.rateFunction = rateFunction
        self.voltageDiffEq = voltageDiffEq

    def nextVoltagesAndRates(self,previousVoltages,previousRates,externalInput,eulerTimeLength):
        diffEq = self.voltageDiffEq(self.weightMatrix,previousRates,previousVoltages,externalInput)
        nextNetworkVoltage = previousVoltages + diffEq*eulerTimeLength
        nextNetworkRates = self.rateFunction(nextNetworkVoltage)
        return nextNetworkVoltage,nextNetworkRates
    
    def eulerSimulate(self,inputTimeMatrix,initialVoltages,eulerTimeLength=.001,progressBar=False):
        timeSteps = inputTimeMatrix.shape[1]
        #initializes the data to be collected
        voltagesData = np.zeros(shape=(self.networkSize,timeSteps))
        ratesData = np.zeros(shape=(self.networkSize,timeSteps))
        #initializes the voltages and rates
        voltages = initialVoltages
        rates = self.rateFunction(initialVoltages)
        #performs the simulation
        if progressBar:
            iterator = tqdm(range(timeSteps))
        else:
            iterator = range(timeSteps)
        for step in iterator:
            externalInput = np.expand_dims(inputTimeMatrix.T[step],axis=1)
            voltages,rates = self.nextVoltagesAndRates(voltages,rates,externalInput,eulerTimeLength)
            voltagesData[:,step] = np.squeeze(voltages)
            ratesData[:,step] = np.squeeze(rates)
        return voltagesData,ratesData