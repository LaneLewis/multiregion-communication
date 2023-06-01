import pandas as pd
import numpy as np
from plotting.colors import colorMappingToColors

def plotConstantInputAcrossTime(inputDF,inputLabel,ax,colorMapping,
                                title='Input Current Over Time',xTicks='implied',
                                yTicks='implied',showFig=True,xlabel='Time',ylabel='Input Current'):
    inputSeries = pd.Series(inputDF[inputLabel],index=inputDF.index)
    inputSeries.plot(color = colorMapping[inputLabel],ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not xTicks == 'implied':
        ax.set_xticks(xTicks)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)   

def plotRateAcrossTime(rateDF,ax,colorMapping,title='Voltage Over Time',
                        xTicks='implied',yTicks='implied',showFig=True,xlabel='',ylabel='Firing Rate'):
    rateDF.plot(color = colorMappingToColors(rateDF,colorMapping),ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not xTicks == 'implied':
        ax.set_xticks(xTicks)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)  

def plotVoltageAcrossTime(voltageDF,ax,colorMapping,title='Voltage Over Time',
                        xTicks='implied',yTicks='implied',showFig=True,xlabel='Time',ylabel='Voltage'):
    voltageDF.plot(color = colorMappingToColors(voltageDF,colorMapping),ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not xTicks == 'implied':
        ax.set_xticks(xTicks)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)