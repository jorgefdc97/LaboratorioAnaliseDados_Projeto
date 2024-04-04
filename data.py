import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("GOOG.US_D1_cleaned.csv")

df_OV = df.loc[:,'open':'volume'] 
df_rsi = df.loc[:,'rsi_3']
df_stoch = df.loc[:,'stoch_3_6_slowk']
df_sr = df.loc[:,'stochrsi_3_6_fastk']
df_mom = df.loc[:,'mom_3']
df_obv = df.loc[:,'obv_0']
df_bb = df.loc[:,'bbands_3_upperband':'bbands_3_lowerband']
df_ema = df.loc[:,'ema_3']
df_sma = df.loc[:,'sma_3']

df_all = pd.concat([df_OV,df_rsi,df_stoch,df_sr,df_mom,df_obv,df_bb,df_ema,df_sma],axis=1)

print('Df:\n',df_all)
print('Description:\n',df_all.describe())
print('Covariance:\n',df_all['volume'].cov(df_all['mom_3']))
print('Correlation:\n',df_all.corr())

plt.scatter(df['volume'],df['mom_3'])
plt.show() 

plt.bar(df['open'],df['obv_0'])
plt.show()

sns.heatmap(df_all.corr())
plt.show() 


