import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("GOOG.US_D1_cleaned.csv")

stoch = df.loc[:,'open':'volume'] 

print('Description:\n',stoch.describe())
print('Covariance:\n',stoch.cov())
print('Correlation:\n',stoch.corr())

plt.scatter(stoch['open'],stoch['volume'])
plt.show() 

plt.bar(stoch['open'],stoch['volume'])
plt.show() 

sns.heatmap(stoch.corr())
plt.show() 


