import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("GOOG.US_D1_cleaned.csv")





stoch = df.loc[:,'open':'volume'] 

print('Description:\n',stoch.describe())
print('Cocariance:\n',stoch.cov())
print('Correlation:\n',stoch.corr())

sns.heatmap(stoch.corr())
plt.show() 


