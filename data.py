import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import GaussianNB
# Read the dataset
df = pd.read_csv("GOOG.US_D1_cleaned.csv")

# Feature selection
df_OV = df.loc[:, 'open':'volume']
df_rsi = df.loc[:, 'rsi_3']
df_stoch = df.loc[:, 'stoch_3_6_slowk']
df_sr = df.loc[:, 'stochrsi_3_6_fastk']
df_mom = df.loc[:, 'mom_3']
df_will = df.loc[:, 'willr_3']
df_obv = df.loc[:, 'obv_0']
df_bb = df.loc[:, ['bbands_3_upperband', 'bbands_3_lowerband']]
df_ema = df.loc[:, 'ema_3']
df_sma = df.loc[:, 'sma_3']

# Concatenate all features
df_all = pd.concat([df_OV, df_rsi, df_stoch, df_sr, df_mom, df_will, df_obv, df_bb, df_ema, df_sma], axis=1)

# Selection of the lines from 3 mounths before the russian war
df_mean_3b = df_all[1545:1608]

# Selection of the lines from 3 mounths after the russian war
df_mean_3a = df_all[1609:1682]
'''
# Describe
pd.set_option('display.max_columns', None)
print('Description:\n', df_mean_3b.describe())
print('Description:\n', df_mean_3a.describe())
print('Covariance:\n', df_all.cov())
print('Correlation:\n', df_all.corr())

# Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df_all.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot showing positive correlation
plt.scatter(df['high'],df['bbands_3_upperband'])
plt.show()

# Histogram showing occurence of transaction volumes
hist = df_all.loc[:,['volume']]
plt.hist(hist)
plt.gca().set(title = 'Histogram',xlabel = 'volume',ylabel = 'number of days')
plt.show()

# Plotting high values over time
plt.figure(figsize=(10, 6))
plt.plot(df["high"])
plt.title('Highest value of stock sold by day')
plt.xlabel('Date')
plt.ylabel('High value')
plt.xticks(rotation=45)
plt.show()

# Plotting volume over time
plt.figure(figsize=(10, 6))
plt.plot(df['volume'], color='green')
plt.title('Volume of stock sold by day')
plt.xlabel('Date')
plt.ylabel('Volume sold')
plt.xticks(rotation=45)
plt.show()

# Plotting rsi values over time
plt.figure(figsize=(10, 6))
plt.plot(df["rsi_3"])
plt.title('RSI by day')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.xticks(rotation=45)
plt.show()

# Plotting momentum values over time
plt.figure(figsize=(10, 6))
plt.plot(df["mom_3"])
plt.title('Momentum by day')
plt.xlabel('Date')
plt.ylabel('Momentum')
plt.xticks(rotation=45)
plt.show()

# Plotting high,low,open,close
plt.figure(figsize=(10, 6))
plt.plot(df['high'], label='High', linestyle='--')
plt.plot(df['low'], label='Low', linestyle='--', color='green')
plt.plot(df['open'], label='Open',  color='purple')
plt.plot(df['close'], label='Close', color='red')
plt.title('Different stock prices over the day')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
'''
# Linear Regression
X=df_all.iloc[:, 1:]
Y=df_all['open']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("The model performance using score:")
print("--------------------------------------")
print(lr.score(X_test,Y_test))
y_test_predict = lr.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)
print("The model performance using R2")
print("--------------------------------------")
print('R2 score is {}'.format(r2))

# Naive Bayes 
def rating_function(x,y):
    if x >= y:
        return 1
    else:
        return 0

df["drop"] =  df[['open','close']].apply(rating_function)

X=df_all.iloc[:, :]
Y=df_all['drop']
X2_train, X2_test, y2_train, y2_test = train_test_split(X, Y, test_size=0.3, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X2_train, y2_train).predict(X2_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X2_test.shape[0], (y2_test != y_pred).sum()))
print(classification_report(y2_test, y_pred))
cm = confusion_matrix(y2_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

gnb = GaussianNB()
scores = cross_val_score(gnb, X, Y, cv=5)
print("------------------------------------\n",scores)
print('\n Mean:', scores.mean())
print('\n STD:', scores.std())