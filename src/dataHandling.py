import dill as pkl
import numpy as np
import pandas as pd
def loadData(filePath):
    return pkl.load(open(filePath,'rb'))

def saveData(data,filePath):
    pkl.dump(data,open(filePath,'wb'))

def makeVoltageDF(dataset):
    return pd.DataFrame(dataset['voltages'].T,columns=dataset['labels'],index=dataset['time'])

def makeRateDF(dataset):
    return pd.DataFrame(dataset['rates'].T,columns=dataset['labels'],index=dataset['time'])

def makeInputDF(dataset):
    inputData = np.concatenate([dataset['noiseInput'],dataset['input'],dataset['totalInput']],axis=0).T
    return pd.DataFrame(inputData,columns=[*dataset['noiseLabels'],*dataset['inputLabels'],*dataset['totalInputLabels']],index=dataset['time'])