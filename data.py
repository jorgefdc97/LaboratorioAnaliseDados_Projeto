import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay, mean_absolute_error, silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


# Define rating function
def rating_function(open_position):
    return 1 if open_position >= 0 else 0


# Read the dataset
df = pd.read_csv("GOOG.US_D1_cleaned.csv")

# Feature selection
features = ['open', 'high', 'low', 'close', 'volume', 'rsi_3', 'stoch_3_6_slowk', 'stochrsi_3_6_fastk', 'mom_3',
            'willr_3', 'obv_0', 'bbands_3_upperband', 'bbands_3_lowerband', 'ema_3', 'sma_3']
df_all = df[features].copy()
df_all["open_close_diff"] = df_all["open"] - df["close"].shift(1)
df_all.fillna(1, inplace=True)
df_all["open_close"] = df_all["open_close_diff"].apply(rating_function)

# Display initial data
print(df_all.head())

# Selecting data from specific timeframes
df_mean_3b = df_all.iloc[1545:1608]
df_mean_3a = df_all.iloc[1609:1682]

# Descriptive statistics
pd.set_option('display.max_columns', None)
print('Description before war:\n', df_mean_3b.describe())
print('Description after war:\n', df_mean_3a.describe())
print('Covariance:\n', df_all.cov())
print('Correlation:\n', df_all.corr())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_all.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot for 'high' vs 'bbands_3_upperband'
plt.scatter(df['high'], df['bbands_3_upperband'])
plt.xlabel('High')
plt.ylabel('BBands 3 Upper Band')
plt.show()

# Histogram for transaction volumes
plt.hist(df_all['volume'], bins=30)
plt.title('Histogram of Volume')
plt.xlabel('Volume')
plt.ylabel('Number of Days')
plt.show()

# Line plots for various metrics over time
metrics = ['high', 'volume', 'rsi_3', 'mom_3']
titles = ['Highest Value of Stock Sold by Day', 'Volume of Stock Sold by Day', 'RSI by Day', 'Momentum by Day']

for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    plt.plot(df[metric])
    plt.title(titles[i])
    plt.xlabel('Date')
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45)
    plt.show()

# Plotting high, low, open, close values
plt.figure(figsize=(10, 6))
plt.plot(df['high'], label='High', linestyle='--')
plt.plot(df['low'], label='Low', linestyle='--', color='green')
plt.plot(df['open'], label='Open', color='purple')
plt.plot(df['close'], label='Close', color='red')
plt.title('Different Stock Prices Over the Day')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Linear Regression
X = df_all.drop(columns=['open'])
Y = df_all['open']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("Linear Regression Score:", lr.score(X_test, Y_test))
y_test_predict = lr.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(Y_test, y_test_predict)))
print("R2 Score:", r2_score(Y_test, y_test_predict))

# Naive Bayes Classification
X = df_all.drop(columns=['open_close'])
Y = df_all['open_close']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# Cross-validation for Naive Bayes
scores = cross_val_score(gnb, X, Y, cv=5)
print("Cross-validation scores:", scores)
print("Mean:", scores.mean())
print("STD:", scores.std())

# KMeans Clustering
if 'high' in df_all.columns and 'low' in df_all.columns:
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(df_all[['open', 'high', 'low']])
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Fitting KMeans with 5 clusters
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_all['cluster'] = kmeans.fit_predict(df_all[['open', 'high', 'low']])
    centroids = kmeans.cluster_centers_
    score = silhouette_score(df_all[['open', 'high', 'low']], kmeans.labels_)
    print("Cluster centroids:\n", centroids)
    print("Silhouette Score:", score)

    # Plotting clusters
    for i in range(5):
        plt.scatter(df_all[df_all['cluster'] == i]['high'], df_all[df_all['cluster'] == i]['low'], label=f'Cluster {i}')
    plt.scatter(centroids[:, 1], centroids[:, 2], s=300, c='black', marker='x', label='Centroids')
    plt.xlabel('High')
    plt.ylabel('Low')
    plt.legend()
    plt.show()

# Hierarchical clustering
linkage_matrix = linkage(df_all[['open', 'high', 'low']], method='ward')
plt.figure(figsize=(14, 7))
dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# MLPRegressor
X = df_all.drop(columns=['open'])
Y = df_all['open']
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.3, random_state=42)
mlp = MLPRegressor(max_iter=500, random_state=42)
mlp.fit(X_train, Y_train)
print("MLPRegressor R2 Score:", r2_score(Y_test, mlp.predict(X_test)))
print("MLPRegressor MAE:", mean_absolute_error(Y_test, mlp.predict(X_test)))

# PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_all.drop(columns=['open', 'cluster']))
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
print("PCA Transformed Data Shape:", pca_data.shape)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df_all['open'], cmap='prism')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# TruncatedSVD
svd = TruncatedSVD(n_components=2)
svd_data = svd.fit_transform(scaled_data)
print("Original Data Shape:", scaled_data.shape)
print("TruncatedSVD Transformed Data Shape:", svd_data.shape)
plt.scatter(svd_data[:, 0], svd_data[:, 1], c=df_all['open'], cmap='prism')
plt.xlabel('First SVD Component')
plt.ylabel('Second SVD Component')
plt.show()

# Logistic regression with class weights
X = df_all.drop(columns=['open_close'])
Y = df_all['open_close']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
log_reg = LogisticRegression(max_iter=200, class_weight='balanced')
log_reg.fit(X_res, y_res)
y_pred = log_reg.predict(X_test)
# Classification Report
print("Logistic Regression Classification Report with Balanced Class Weights:\n", classification_report(y_test, y_pred))
# ROC-AUC Score
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)
print("Ridge Regression Score:", ridge.score(X_test, Y_test))

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
print("Lasso Regression Score:", lasso.score(X_test, Y_test))

# SVM with different kernels
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print(f"SVM with {kernel} kernel Classification Report:\n", classification_report(y_test, y_pred))
