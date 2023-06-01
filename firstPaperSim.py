import numpy as np
import dill as pkl
from src.voltageRateNetwork import VoltageRateNetwork
from src.OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcessEuler
from src.dataHandling import saveData
from src.rateFunctions import powerRateFunctionConstructor
from src.voltageDiffEqs import regionVoltageDiffEqConstructor
from src.eulerHelpers import EulerVariables

def singleRegionDiffEq(eTau,iTau,restingVoltageE,restingVoltageI,):
    restingVoltages = np.array([[restingVoltageE],
                                [restingVoltageI]])
    voltageTimeConstants = np.array([[eTau],
                                     [iTau]])
    return regionVoltageDiffEqConstructor(restingVoltages=restingVoltages,timeConstants=voltageTimeConstants)

def singleRegionRateEq(voltageThresholdE,voltageThresholdI,ratePower,rateK):
    voltageThresholds = np.array([[voltageThresholdE],
                                  [voltageThresholdI]])    
    return powerRateFunctionConstructor(voltageThresholds=voltageThresholds,power=ratePower,k=rateK)

def singleRegionWeights(eToEWeight,eToIWeight,iToEWeight,iToIWeight):
    return np.array([[eToEWeight,-1*iToEWeight],
                     [eToIWeight,-1*iToIWeight]])

def makeSingleRegionNoise(eNoiseStd,eTau,iNoiseStd,iTau,noiseTau,
                      eulerVariables):
    noiseDeviationE = eNoiseStd*np.sqrt(1+eTau/noiseTau)
    noiseDeviationI = iNoiseStd*np.sqrt(1+iTau/noiseTau)
    noiseCovariance = np.array([[noiseDeviationE**2,0],
                                [0,noiseDeviationI**2]])
    noiseInitialConditions = np.array([[0],
                                      [0]])
    print(noiseCovariance)
    return OrnsteinUhlenbeckProcessEuler(noiseCovariance,noiseTau,noiseInitialConditions,
                                  eulerVariables.timeDelta,eulerVariables.divisions)
def makeSingleRegionSharedInput(amplitude,divisions):
    return np.ones((2,divisions))*amplitude

def makeInitialVoltages(initialVoltageE,initialVoltageI):
    return np.array([[initialVoltageE],
                     [initialVoltageI]])

def constructBasicNetwork(eTau,iTau,eToEWeight,eToIWeight,iToEWeight,iToIWeight,
                    restingVoltageE,restingVoltageI,voltageThresholdE,voltageThresholdI,
                    rateK,ratePower):
    weights = singleRegionWeights(eToEWeight=eToEWeight,eToIWeight=eToIWeight,iToEWeight=iToEWeight,iToIWeight=iToIWeight)
    powerRateEquation = singleRegionRateEq(voltageThresholdE=voltageThresholdE,voltageThresholdI=voltageThresholdI,
                                           ratePower=ratePower,rateK=rateK)
    neuronDiffEq = singleRegionDiffEq(eTau=eTau,iTau=iTau,restingVoltageE=restingVoltageE,restingVoltageI=restingVoltageI)
    network = VoltageRateNetwork(weightMatrix=weights,rateFunction=powerRateEquation,voltageDiffEq=neuronDiffEq)
    return network

def runSingleRegionBasic(eTau=20,iTau=10,eToEWeight=1.25,eToIWeight=1.2,iToEWeight=.65,iToIWeight=.5,
                        restingVoltageE=-70,restingVoltageI=-70,voltageThresholdE=-70,voltageThresholdI=-70,
                        eNoiseStd=.2,iNoiseStd=.1,noiseTau=50,powerRateK=.3,powerRateN=2,inputAmplitude=0,
                        eulerTimeStart=0,eulerTimeEnd=2500,initialVoltageE=-70,initialVoltageI=-70,divisions=10000,
                        saveName='singleRegionBasic'):
    parameters = locals()
    #constructs the network with the proper parameters
    network = constructBasicNetwork(eTau=eTau,iTau=iTau,eToEWeight=eToEWeight,eToIWeight=eToIWeight,
                                    iToEWeight=iToEWeight,iToIWeight=iToIWeight,restingVoltageE=restingVoltageE,
                                    restingVoltageI=restingVoltageI,voltageThresholdE=voltageThresholdE,
                                    voltageThresholdI=voltageThresholdI,rateK=powerRateK,ratePower=powerRateN)
    #constructs the simulation parameters
    eulerVariables = EulerVariables(eulerTimeStart,eulerTimeEnd,divisions)
    inputNoiseTimeSeries = makeSingleRegionNoise(eNoiseStd,eTau,iNoiseStd,iTau,noiseTau,eulerVariables)
    constantInputTimeSeries = makeSingleRegionSharedInput(inputAmplitude,divisions)
    totalInputTimeSeries = inputNoiseTimeSeries+constantInputTimeSeries
    initialVoltageConditions = makeInitialVoltages(initialVoltageE,initialVoltageI)
    simulationVoltages,simulationRates = network.eulerSimulate(totalInputTimeSeries,initialVoltageConditions,eulerVariables.timeDelta)
    #saves the data
    dataOut = {'params':parameters,'voltages':simulationVoltages,'rates':simulationRates,'labels':['E','I'],
               'time':eulerVariables.time,'noiseInput':inputNoiseTimeSeries,'input':constantInputTimeSeries,
               'totalInput':totalInputTimeSeries,
               'noiseLabels':['eNoise','iNoise'],
               'inputLabels':['eInput','iInput'],
               'totalInputLabels':['eTotalInput','iTotalInput']}
    saveData(dataOut,f'./singleRegion/data/{saveName}.pkl')

def runSingleRegionMultiInput(eTau=20,iTau=10,eToEWeight=1.25,eToIWeight=1.2,iToEWeight=.65,iToIWeight=.5,
                        restingVoltageE=-70,restingVoltageI=-70,voltageThresholdE=-70,voltageThresholdI=-70,
                        eNoiseStd=.2,iNoiseStd=.1,noiseTau=50,rateK=.3,ratePower=2,inputAmplitudes=[0],
                        eulerTimeStart=0,eulerTimeEnd=75000,divisions=750000,initialVoltageE=-70,initialVoltageI=-70,
                        saveName='singleRegionMultiInput'):
    parameters = locals()
    #constructs the network with the proper parameters
    network = constructBasicNetwork(eTau=eTau,iTau=iTau,eToEWeight=eToEWeight,eToIWeight=eToIWeight,
                                    iToEWeight=iToEWeight,iToIWeight=iToIWeight,restingVoltageE=restingVoltageE,
                                    restingVoltageI=restingVoltageI,voltageThresholdE=voltageThresholdE,
                                    voltageThresholdI=voltageThresholdI,rateK=rateK,ratePower=ratePower)
    #constructs the euler time parameters
    timeLength = eulerTimeEnd - eulerTimeStart
    totalEulerTimeEnd = len(inputAmplitudes)*timeLength+eulerTimeStart
    totalDivisions = len(inputAmplitudes)*divisions
    #euler seems correct
    eulerVariables = EulerVariables(eulerTimeStart,totalEulerTimeEnd,totalDivisions)
    #inputs seems correct
    inputNoiseTimeSeries = makeSingleRegionNoise(eNoiseStd=eNoiseStd,eTau=eTau,iNoiseStd=iNoiseStd,iTau=iTau,noiseTau=noiseTau,eulerVariables=eulerVariables)
    constantInputTimeSeries = np.concatenate([makeSingleRegionSharedInput(amplitude,divisions) for amplitude in inputAmplitudes],axis=1)
    totalInputTimeSeries = inputNoiseTimeSeries+constantInputTimeSeries

    initialVoltageConditions = makeInitialVoltages(initialVoltageE=initialVoltageE,initialVoltageI=initialVoltageI)
    simulationVoltages,simulationRates = network.eulerSimulate(inputTimeMatrix=totalInputTimeSeries,initialVoltages=initialVoltageConditions,
                                                               eulerTimeLength=eulerVariables.timeDelta)
    #saves the data
    dataOut = {'params':parameters,'voltages':simulationVoltages,'rates':simulationRates,'labels':['E','I'],
               'time':eulerVariables.time/1000,'noiseInput':inputNoiseTimeSeries,'input':constantInputTimeSeries,
               'totalInput':totalInputTimeSeries,
               'noiseLabels':['eNoise','iNoise'],
               'inputLabels':['eInput','iInput'],
               'totalInputLabels':['eTotalInput','iTotalInput']}
    saveData(dataOut,f'./singleRegion/data/{saveName}.pkl')