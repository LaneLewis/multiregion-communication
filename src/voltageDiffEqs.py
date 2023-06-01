import numpy as np

def regionVoltageDiffEqConstructor(restingVoltages,timeConstants):
    def regionVoltageDiffEq(weightMatrix,previousRates,previousVoltages,externalInput):
        rateInput = np.matmul(weightMatrix,previousRates)
        return (-1*previousVoltages + restingVoltages + externalInput + rateInput)/timeConstants
    return regionVoltageDiffEq
