import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("GOOG.US_D1_cleaned.csv")





stoch = df.loc[:,'open':'volume'] 

print('Description:\n',stoch.describe())
sns.heatmap(corr.corr())
plt.show() 


