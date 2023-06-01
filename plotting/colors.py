def colorMappingToColors(df,colorMapping):
    keys = df.columns 
    return [colorMapping[key] for key in keys]