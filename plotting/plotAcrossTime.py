import pandas as pd
import numpy as np
from plotting.colors import colorMappingToColors
from plotting.plotAcrossInput import plotDF
def plotConstantInputAcrossTime(inputDF,inputLabel,ax,colorMapping,
                                title='Input Current Over Time',xTicks='implied',
                                yTicks='implied',showFig=True,xlabel='Time',ylabel='Input Current',
                                lineWidth=.5):
    inputSeries = pd.Series(inputDF[inputLabel],index=inputDF.index)
    line=inputSeries.plot(color = colorMapping[inputLabel],ax=ax,linewidth=lineWidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not xTicks == 'implied':
        ax.set_xticks(xTicks)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)   
    return line

def plotRateAcrossTime(rateDF,ax,colorMapping,title='Voltage Over Time',
                        xTicks='implied',yTicks='implied',showFig=True,xlabel='',ylabel='Firing Rate',lineWidth=.5):
    lines = plotDF(rateDF,colors=colorMappingToColors(rateDF,colorMapping),axis=ax,lineWidth=lineWidth,marker=None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not xTicks == 'implied':
        ax.set_xticks(xTicks)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)  
    return lines

def plotVoltageAcrossTime(voltageDF,ax,colorMapping,title='Voltage Over Time',
                        xTicks='implied',yTicks='implied',showFig=True,xlabel='Time',ylabel='Voltage',lineWidth=.5):
    lines=plotDF(voltageDF,colors = colorMappingToColors(voltageDF,colorMapping),axis=ax,lineWidth=lineWidth,marker=None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if not xTicks == 'implied':
        ax.set_xticks(xTicks)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)
    return lines