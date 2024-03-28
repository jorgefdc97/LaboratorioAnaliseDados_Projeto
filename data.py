import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("GOOG.US_D1_cleaned.csv")

print('Description:\n',df.describe())



stoch = df.loc[:,'open':'volume'] 
rsi = df.loc[:,'rsi_3':'rsi_90']


df_corr = pd.concat([stoch,rsi], axis=1)
corr = df_corr[df_corr['volume'] > 50000000]

sns.heatmap(corr.corr())
plt.show() 


