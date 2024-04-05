import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("GOOG.US_D1_cleaned.csv")

df_OV = df.loc[:,'open':'volume'] 
df_rsi = df.loc[:,'rsi_3']
df_stoch = df.loc[:,['stoch_3_6_slowk','stoch_3_6_slowd']]
df_sr = df.loc[:,['stochrsi_3_6_fastk','stochrsi_3_6_fastd']]
df_mom = df.loc[:,'mom_3']
df_will = df.loc[:,'willr_3']
df_obv = df.loc[:,'obv_0']
df_bb = df.loc[:,['bbands_3_upperband','bbands_3_lowerband']]
df_ema = df.loc[:,'ema_3']
df_sma = df.loc[:,'sma_3']

df_all = pd.concat([df_OV,df_rsi,df_stoch,df_sr,df_mom,df_will,df_obv,df_bb,df_ema,df_sma],axis=1)

 

print('Df:\n',df_all)
print('Description:\n',df_all.describe())
print('Covariance:\n',df_all.cov()) 
'''df_all['high'].cov(df_all['bbands_3_upperband'])'''
print('Correlation:\n',df_all.corr())

plt.scatter(df['high'],df['bbands_3_upperband'])
plt.show() 

sns.heatmap(df_all.corr(),annot=True)
plt.show() 

hist = df_all.loc[:,['high','bbands_3_upperband']]


plt.hist(hist)
plt.gca().set(title = 'Histogram',xlabel = 'high',ylabel = 'bbands_3_upperband')
plt.show() 

'''plt.plot(df["datetime"],df["high"])'''
plt.plot(df["high"])
plt.gca().set(title = 'Highest value of stock sold by day number',xlabel = 'days',ylabel = 'value')
plt.show()

'''plt.plot(df["datetime"],df['volume'])'''
plt.plot(df['volume'])
plt.gca().set(title = 'volume of stock sold by day number',xlabel ='days',ylabel = 'volume sold')
plt.show()

