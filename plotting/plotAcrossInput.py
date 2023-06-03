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
                                  yTicks='implied',xlabel='Input Current',ylabel='Mean Voltage',legend=False,lineWidth=1):
    dfs,inputs = splitFrameAcrossInput(voltageFrame,inputDF,inputLabel)
    averageFrame = pd.DataFrame([df.mean() for df in dfs],index=inputs)
    lines=plotDF(averageFrame,colors=colorMappingToColors(averageFrame,colorMapping),axis=ax,lineWidth=lineWidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(inputs)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)   
    return lines

def plotStdVoltageAcrossInput(voltageFrame,inputDF,ax,colorMapping,inputLabel,
                              title='Average Voltage Response to Input Current',
                              yTicks='implied',xlabel='InputCurrent',ylabel='Voltage Std',legend=False,lineWidth=1):
    dfs,inputs = splitFrameAcrossInput(voltageFrame,inputDF,inputLabel)
    averageFrame = pd.DataFrame([df.std() for df in dfs],index=inputs)
    lines = plotDF(averageFrame,colors=colorMappingToColors(averageFrame,colorMapping),axis=ax,lineWidth=lineWidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(inputs)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)
    return lines

def plotAverageRateAcrossInput(rateFrame,inputDF,ax,colorMapping,inputLabel,title='Average Rate Response to Input Current',
                                yTicks='implied',xlabel='Input Current',ylabel='Mean Rate',lineWidth=1):
    dfs,inputs = splitFrameAcrossInput(rateFrame,inputDF,inputLabel)
    averageFrame = pd.DataFrame([df.mean() for df in dfs],index=inputs)
    lines = plotDF(averageFrame,colors=colorMappingToColors(averageFrame,colorMapping),axis=ax,lineWidth=lineWidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(inputs)
    if not yTicks == 'implied':
        ax.set_yticks(yTicks)  
    return lines

def plotDF(frame,colors,axis,marker='o',lineWidth=1):
    labels= frame.columns
    lines = []
    for i in range(len(frame.T)):
        line,=axis.plot(frame.iloc[:,i],color=colors[i],marker=marker,label=labels[i],linewidth=lineWidth)
        lines.append(line)
    return lines
    