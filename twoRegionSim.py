import numpy as np
import dill as pkl
from src.voltageRateNetwork import VoltageRateNetwork
from src.OrnsteinUhlenbeckProcess import OrnsteinUhlenbeckProcessEuler
from src.dataHandling import saveData
from src.rateFunctions import powerRateFunctionConstructor
from src.voltageDiffEqs import regionVoltageDiffEqConstructor
from src.eulerHelpers import EulerVariables
from src.dataHandling import loadData, makeVoltageDF,makeRateDF,makeInputDF,saveData
from plotting.plotAcrossTime import plotConstantInputAcrossTime,plotVoltageAcrossTime,plotRateAcrossTime
from plotting.plotAcrossInput import plotAverageVoltageAcrossInput,plotStdVoltageAcrossInput,splitFrameAcrossInput#ordering R1E,R1I,R2E,R2I

def twoRegionDiffEq(r1ETau,r1ITau,r2ETau,r2ITau,
                    r1ERestingVoltage,r1IRestingVoltage,
                    r2ERestingVoltage,r2IRestingVoltage):
    restingVoltages = np.array([[r1ERestingVoltage],
                                [r1IRestingVoltage],
                                [r2ERestingVoltage],
                                [r2IRestingVoltage]])
    voltageTimeConstants = np.array([[r1ETau],
                                     [r1ITau],
                                     [r2ETau],
                                     [r2ITau]])
    return regionVoltageDiffEqConstructor(restingVoltages=restingVoltages,timeConstants=voltageTimeConstants)

def twoRegionRateEq(r1EVoltageThreshold,r1IVoltageThreshold,
                    r2EVoltageThreshold,r2IVoltageThreshold,
                    ratePower,rateK):
    voltageThresholds = np.array([[r1EVoltageThreshold],
                                  [r1IVoltageThreshold],
                                  [r2EVoltageThreshold],
                                  [r2IVoltageThreshold]]) 
       
    return powerRateFunctionConstructor(voltageThresholds=voltageThresholds,power=ratePower,k=rateK)

def twoRegionWeights(r1EToR1EWeight,r1EToR1IWeight,r1IToR1IWeight,r1IToR1EWeight,
                     r2EToR2EWeight,r2EToR2IWeight,r2IToR2IWeight,r2IToR2EWeight,
                     r1EToR2EWeight,r1EToR2IWeight,r1IToR2EWeight,r1IToR2IWeight,
                     r2EToR1EWeight,r2EToR1IWeight,r2IToR1EWeight,r2IToR1IWeight):
    return np.array([[r1EToR1EWeight,-1*r1IToR1EWeight,r2EToR1EWeight,-1*r2IToR1EWeight],
                     [r1EToR1IWeight,-1*r1IToR1IWeight,r2EToR1IWeight,-1*r2IToR1IWeight],
                     [r1EToR2EWeight,-1*r1IToR2EWeight,r2EToR2EWeight,-1*r2IToR2EWeight],
                     [r1EToR2IWeight,-1*r1IToR2IWeight,r2EToR2IWeight,-1*r2IToR2IWeight]])

def makePrivateTwoRegionNoise(r1ENoiseStd,r1INoiseStd,r2ENoiseStd,r2INoiseStd,
                       r1ETau,r1ITau,r2ETau,r2ITau,
                       noiseTau,eulerVariables):
    #covariance is correct
    r1ENoiseDeviation = r1ENoiseStd*np.sqrt(1+r1ETau/noiseTau)
    r1INoiseDeviation = r1INoiseStd*np.sqrt(1+r1ITau/noiseTau)
    r2ENoiseDeviation = r2ENoiseStd*np.sqrt(1+r2ETau/noiseTau)
    r2INoiseDeviation = r2INoiseStd*np.sqrt(1+r2ITau/noiseTau)
    noiseCovariance = np.diag(np.square([r1ENoiseDeviation,r1INoiseDeviation,r2ENoiseDeviation,r2INoiseDeviation]))
    noiseInitialConditions = np.zeros(shape=(4,1))
    return OrnsteinUhlenbeckProcessEuler(noiseCovariance,noiseTau,noiseInitialConditions,
                                  eulerVariables.timeDelta,eulerVariables.divisions)

def makeTwoRegionInput(r1Amplitude,r2Amplitude,divisions):
    return np.concatenate([np.ones((2,divisions))*r1Amplitude,np.ones((2,divisions))*r2Amplitude],axis=0)

def makeTwoRegionInitialVoltages(r1EInitialVoltage,r1IInitialVoltage,
                                r2EInitialVoltage,r2IInitialVoltage):
    return np.array([[r1EInitialVoltage],
                     [r1IInitialVoltage],
                     [r2EInitialVoltage],
                     [r2IInitialVoltage]])

def constructTwoRegionNetwork(r1ETau,r1ITau,r2ETau,r2ITau,
                             r1ERestingVoltage,r1IRestingVoltage,
                             r2ERestingVoltage,r2IRestingVoltage,
                             r1EVoltageThreshold,r1IVoltageThreshold,
                             r2EVoltageThreshold,r2IVoltageThreshold,
                             ratePower,rateK,
                             r1EToR1EWeight,r1EToR1IWeight,r1IToR1IWeight,r1IToR1EWeight,
                             r2EToR2EWeight,r2EToR2IWeight,r2IToR2IWeight,r2IToR2EWeight,
                             r1EToR2EWeight,r1EToR2IWeight,r1IToR2EWeight,r1IToR2IWeight,
                             r2EToR1EWeight,r2EToR1IWeight,r2IToR1EWeight,r2IToR1IWeight):
    
    weights = twoRegionWeights(r1EToR1EWeight=r1EToR1EWeight,r1EToR1IWeight=r1EToR1IWeight,r1IToR1IWeight=r1IToR1IWeight,
                               r1IToR1EWeight=r1IToR1EWeight,r2EToR2EWeight=r2EToR2EWeight,r2EToR2IWeight=r2EToR2IWeight,
                               r2IToR2IWeight=r2IToR2IWeight,r2IToR2EWeight=r2IToR2EWeight,r1EToR2EWeight=r1EToR2EWeight,
                               r1EToR2IWeight=r1EToR2IWeight,r1IToR2EWeight=r1IToR2EWeight,r1IToR2IWeight=r1IToR2IWeight,
                               r2EToR1EWeight=r2EToR1EWeight,r2EToR1IWeight=r2EToR1IWeight,r2IToR1EWeight=r2IToR1EWeight,
                               r2IToR1IWeight=r2IToR1IWeight)
    powerRateEquation = twoRegionRateEq(r1EVoltageThreshold=r1EVoltageThreshold,r1IVoltageThreshold=r1IVoltageThreshold,
                                        r2EVoltageThreshold=r2EVoltageThreshold,r2IVoltageThreshold=r2IVoltageThreshold,
                                        ratePower=ratePower,rateK=rateK)
    neuronDiffEq = twoRegionDiffEq(r1ETau=r1ETau,r1ITau=r1ITau,r2ETau=r2ETau,r2ITau=r2ITau,
                                   r1ERestingVoltage=r1ERestingVoltage,r1IRestingVoltage=r1IRestingVoltage,
                                   r2ERestingVoltage=r2ERestingVoltage,r2IRestingVoltage=r2IRestingVoltage)
    network = VoltageRateNetwork(weightMatrix=weights,rateFunction=powerRateEquation,voltageDiffEq=neuronDiffEq)
    return network

def twoRegionMultiInput(
                r1ETau=20,r1ITau=10,r2ETau=20,r2ITau=10,
                r1ERestingVoltage=-70,r1IRestingVoltage=-70,r2ERestingVoltage=-70,r2IRestingVoltage=-70,
                r1EVoltageThreshold=-70,r1IVoltageThreshold=-70,r2EVoltageThreshold=-70,r2IVoltageThreshold=-70,
                ratePower=2,rateK=.3,
                r1EToR1EWeight=1.25,r1EToR1IWeight=1.2,r1IToR1IWeight=.5,r1IToR1EWeight=.65,
                r2EToR2EWeight=1.25,r2EToR2IWeight=1.2,r2IToR2IWeight=.5,r2IToR2EWeight=.65,
                r1EToR2EWeight=0,r1EToR2IWeight=0,r1IToR2EWeight=0,r1IToR2IWeight=0,
                r2EToR1EWeight=0,r2EToR1IWeight=0,r2IToR1EWeight=0,r2IToR1IWeight=0,
                r1EInitialVoltage=-70,r1IInitialVoltage=-70,r2EInitialVoltage=-70,r2IInitialVoltage=-70,
                r1ENoiseStd=.2,r1INoiseStd=.1,r2ENoiseStd=.2,r2INoiseStd=.1,noiseTau=50,
                r1InputAmplitudes=[0],r2InputAmplitudes=[0],
                eulerTimeStart=0,eulerTimeEnd=75000,divisions=750000,
                saveName='twoRegionMultiInput',progressBar=False):
    assert len(r1InputAmplitudes) == len(r2InputAmplitudes)
    inputLength= len(r1InputAmplitudes)
    parameters = locals()
    #constructs the variables for euler integration
    timeLength = eulerTimeEnd - eulerTimeStart
    totalEulerTimeEnd = inputLength*timeLength+eulerTimeStart
    totalDivisions = inputLength*divisions
    eulerVariables = EulerVariables(eulerTimeStart,totalEulerTimeEnd,totalDivisions)
    #constructs the network with the proper parameters
    network = constructTwoRegionNetwork(r1ETau=r1ETau,r1ITau=r1ITau,r2ETau=r2ETau,r2ITau=r2ITau,
                             r1ERestingVoltage=r1ERestingVoltage,r1IRestingVoltage=r1IRestingVoltage,
                             r2ERestingVoltage=r2ERestingVoltage,r2IRestingVoltage=r2IRestingVoltage,
                             r1EVoltageThreshold=r1EVoltageThreshold,r1IVoltageThreshold=r1IVoltageThreshold,
                             r2EVoltageThreshold=r2EVoltageThreshold,r2IVoltageThreshold=r2IVoltageThreshold,
                             ratePower=ratePower,rateK=rateK,
                             r1EToR1EWeight=r1EToR1EWeight,r1EToR1IWeight=r1EToR1IWeight,
                             r1IToR1IWeight=r1IToR1IWeight,r1IToR1EWeight=r1IToR1EWeight,
                             r2EToR2EWeight=r2EToR2EWeight,r2EToR2IWeight=r2EToR2IWeight,
                             r2IToR2IWeight=r2IToR2IWeight,r2IToR2EWeight=r2IToR2EWeight,
                             r1EToR2EWeight=r1EToR2EWeight,r1EToR2IWeight=r1EToR2IWeight,
                             r1IToR2EWeight=r1IToR2EWeight,r1IToR2IWeight=r1IToR2IWeight,
                             r2EToR1EWeight=r2EToR1EWeight,r2EToR1IWeight=r2EToR1IWeight,
                             r2IToR1EWeight=r2IToR1EWeight,r2IToR1IWeight=r2IToR1IWeight)
    #constructs the simulation parameters
    inputNoiseTimeSeries = makePrivateTwoRegionNoise(r1ENoiseStd,r1INoiseStd,r2ENoiseStd,r2INoiseStd,
                                                     r1ETau,r1ITau,r2ETau,r2ITau,
                                                     noiseTau,eulerVariables)
    #input noise is correct
    constantInputTimeSeries = np.concatenate([makeTwoRegionInput(r1InputAmplitudes[i],r2InputAmplitudes[i],divisions) for i in range(inputLength)],axis=1)
    totalInputTimeSeries = inputNoiseTimeSeries+constantInputTimeSeries
    initialVoltageConditions = makeTwoRegionInitialVoltages(r1EInitialVoltage,r1IInitialVoltage,
                                                            r2EInitialVoltage,r2IInitialVoltage)
    simulationVoltages,simulationRates = network.eulerSimulate(totalInputTimeSeries,initialVoltageConditions,eulerVariables.timeDelta,progressBar=progressBar)
    #saves the data 
    dataOut = {'params':parameters,'voltages':simulationVoltages,'rates':simulationRates,'labels':['r1E','r1I','r2E','r2I'],
               'time':eulerVariables.time/1000,'noiseInput':inputNoiseTimeSeries,'input':constantInputTimeSeries,
               'totalInput':totalInputTimeSeries,
               'noiseLabels':['r1ENoise','r1INoise','r2ENoise','r2INoise'],
               'inputLabels':['r1EInput','r1IInput','r2EInput','r2IInput'],
               'totalInputLabels':['r1ETotalInput','r1ITotalInput','r2ETotalInput','r2ITotalInput']}
    saveData(dataOut,f'./twoRegion/data/{saveName}.pkl')