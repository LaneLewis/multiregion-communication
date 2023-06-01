import numpy as np
def powerRateFunctionConstructor(voltageThresholds,power,k):
    def powerRateFunction(voltages):
        voltageDelta = voltages - voltageThresholds
        zeros = np.zeros(shape=(len(voltages),1))
        rate = k*np.maximum(voltageDelta,zeros)**power
        return rate
    return powerRateFunction