from plotting.colors import colorMappingToColors
import pandas as pd
def splitFrameAcrossInput(dataframe,inputDF,inputLabel):
    outFrames = []
    uniqueStims = list(set(inputDF[inputLabel]))
    for stim in uniqueStims:
        subsetDF = dataframe[inputDF[inputLabel] == stim]
        outFrames.append(subsetDF)
    return outFrames,uniqueStims

def plotAverageVoltageAcrossInput(voltageFrame,inputDF,ax,colorMapping,inputLabel,title='Average Voltage Response to Input Current',
                                  yTicks='implied',xlabel='Input Current',ylabel='Mean Voltage'):
    dfs,inputs = splitFrameAcrossInput(voltageFrame,inputDF,inputLabel)
    averageFrame = pd.DataFrame([df.mean() for df in dfs],index=inputs)
    averageFrame.plot(color=colorMappingToColors(averageFrame,colorMapping),marker='o',ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(inputs)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)   

def plotStdVoltageAcrossInput(voltageFrame,inputDF,ax,colorMapping,inputLabel,
                              title='Average Voltage Response to Input Current',
                              yTicks='implied',xlabel='InputCurrent',ylabel='Voltage Std'):
    dfs,inputs = splitFrameAcrossInput(voltageFrame,inputDF,inputLabel)
    averageFrame = pd.DataFrame([df.std() for df in dfs],index=inputs)
    averageFrame.plot(color=colorMappingToColors(averageFrame,colorMapping),marker='o',ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(inputs)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)  