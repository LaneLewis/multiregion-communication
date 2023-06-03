import numpy as np
def OrnsteinUhlenbeckProcessEuler(noiseCovariance,timeConstant,initialCondition,
                                  timeDelta=.001,timeSteps=500):
    #uses the euler-maruyama method
    dimension = noiseCovariance.shape[0]
    xs = np.zeros((dimension,timeSteps))
    xs[:,0] = np.squeeze(initialCondition)
    #generates the wiener process normals for the entire sim
    normalSimulation = np.random.normal(0,1,size=(dimension,timeSteps))
    #constant term that gets multiplied by the weiner process
    noiseMult = np.sqrt(2*timeConstant)*matrixSqrt(noiseCovariance)
    for i in range(1,timeSteps):
        normalSample = np.expand_dims(normalSimulation[:,i],axis=1)
        #the stochastic component
        noise = np.squeeze(np.matmul(noiseMult,normalSample))
        xs[:,i] = xs[:,i-1] + (-1*xs[:,i-1]/timeConstant)*timeDelta + noise*np.sqrt(timeDelta)/timeConstant
    return xs

def matrixSqrt(matrix):
    eigs,S = np.linalg.eigh(matrix)
    out = np.matmul(S,np.diag(np.sqrt(eigs)))
    root = np.matmul(out,S.T)
    return root
