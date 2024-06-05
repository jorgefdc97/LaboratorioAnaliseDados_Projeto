import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, \
    ConfusionMatrixDisplay, mean_absolute_error, silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from imblearn.over_sampling import SMOTE
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

def read_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    features = ['open', 'high', 'low', 'close', 'volume', 'rsi_3', 'stoch_3_6_slowk', 'stochrsi_3_6_fastk', 'mom_3',
                'willr_3', 'obv_0', 'bbands_3_upperband', 'bbands_3_lowerband', 'ema_3', 'sma_3']
    df_all = df[features].copy()
    df_all["open_close_diff"] = df_all["open"] - df["close"].shift(1)
    df_all.fillna(1, inplace=True)
    df_all["open_close"] = df_all["open_close_diff"].apply(lambda x: 1 if x >= 0 else 0)
    return df_all


def descriptive_statistics(df, df_mean_3b, df_mean_3a):
    pd.set_option('display.max_columns', None)
    print('Description before period:\n', df_mean_3b.describe())
    print('Description after period:\n', df_mean_3a.describe())
    print('Covariance:\n', df.cov())
    print('Correlation:\n', df.corr())


def correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()


def scatter_plot(df):
    plt.scatter(df['high'], df['bbands_3_upperband'])
    plt.xlabel('High')
    plt.ylabel('BBands 3 Upper Band')
    plt.show()


def histogram(df):
    plt.hist(df['volume'], bins=30)
    plt.title('Histogram of Volume')
    plt.xlabel('Volume')
    plt.ylabel('Number of Days')
    plt.show()


def line_plots(df):
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


def price_plot(df):
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


def linear_regression(df, columns):
    X = df.drop(columns)
    Y = df[columns]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    print("Linear Regression Score:", lr.score(X_test, Y_test))
    y_test_predict = lr.predict(X_test)
    print("RMSE:", np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    print("R2 Score:", r2_score(Y_test, y_test_predict))


def naive_bayes_classification(df, columns):
    X = df.drop(columns)
    Y = df[columns]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()
    scores = cross_val_score(gnb, X, Y, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean:", scores.mean())
    print("STD:", scores.std())


def kmeans_clustering(df):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(df[['open', 'high', 'low']])
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['open', 'high', 'low']])
    centroids = kmeans.cluster_centers_
    score = silhouette_score(df[['open', 'high', 'low']], kmeans.labels_)
    print("Cluster centroids:\n", centroids)
    print("Silhouette Score:", score)
    for i in range(5):
        plt.scatter(df[df['cluster'] == i]['high'], df[df['cluster'] == i]['low'], label=f'Cluster {i}')
    plt.scatter(centroids[:, 1], centroids[:, 2], s=300, c='black', marker='x', label='Centroids')
    plt.xlabel('High')
    plt.ylabel('Low')
    plt.legend()
    plt.show()


def hierarchical_clustering(df):
    linkage_matrix = linkage(df[['open', 'high', 'low']], method='ward')
    plt.figure(figsize=(14, 7))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()


def mlp_regressor(df, columns):
    X = df.drop(columns)
    Y = df[columns]
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.3, random_state=42)
    mlp = MLPRegressor(max_iter=500, random_state=42)
    mlp.fit(X_train, Y_train)
    print("MLPRegressor R2 Score:", r2_score(Y_test, mlp.predict(X_test)))
    print("MLPRegressor MAE:", mean_absolute_error(Y_test, mlp.predict(X_test)))


def pca_analysis(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['open', 'cluster']))
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    print("PCA Transformed Data Shape:", pca_data.shape)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['open'], cmap='prism')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


def truncated_svd_analysis(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['open', 'cluster']))
    svd = TruncatedSVD(n_components=2)
    svd_data = svd.fit_transform(scaled_data)
    print("Original Data Shape:", scaled_data.shape)
    print("TruncatedSVD Transformed Data Shape:", svd_data.shape)
    plt.scatter(svd_data[:, 0], svd_data[:, 1], c=df['open'], cmap='prism')
    plt.xlabel('First SVD Component')
    plt.ylabel('Second SVD Component')
    plt.show()


def logistic_regression_with_class_weights(df, columns):
    X = df.drop(columns)
    Y = df[columns]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    log_reg = LogisticRegression(max_iter=200, class_weight='balanced')
    log_reg.fit(X_res, y_res)
    y_pred = log_reg.predict(X_test)
    print("Logistic Regression Classification Report with Balanced Class Weights:\n",
          classification_report(y_test, y_pred))
    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {roc_auc:.2f}")


def ridge_regression(df, columns):
    X = df.drop(columns)
    Y = df[columns]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    print("Ridge Regression Score:", ridge.score(X_test, Y_test))


def lasso_regression(df, columns):
    X = df.drop(columns)
    Y = df[columns]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, Y_train)
    print("Lasso Regression Score:", lasso.score(X_test, Y_test))


def svm_regression(df, columns):
    X = df.drop(columns)
    Y = df[columns]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    kernels = ['poly', 'rbf']
    for kernel in kernels:
        svm = SVR(kernel=kernel)
        svm.fit(X_train, Y_train)  # Fit SVM regression directly without SMOTE
        y_pred = svm.predict(X_test)
        print(f"SVM with {kernel} kernel R2 Score:", r2_score(Y_test, y_pred))
        print(f"SVM with {kernel} kernel MAE:", mean_absolute_error(Y_test, y_pred))

def time_series_analysis(df):
    # Select relevant columns for time series analysis
    selected_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi_3', 'mom_3']
    df_selected = df[selected_columns].copy()

    # Plot time series for each selected column
    for column in df_selected.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df_selected.index, df_selected[column])
        plt.title(f'Time Series Analysis: {column}')
        plt.xlabel('Line Number')
        plt.ylabel(column.capitalize())
        plt.grid(True)
        plt.show()

    # Statistical summary for each column
    print("Statistical Summary:")
    print(df_selected.describe())

"""
def predict_next_365_days(df):
    # Select relevant column for time series analysis (e.g., 'close' price)
    ts_column = 'close'
    ts_data = df[ts_column]

    # Train ARIMA model
    model = ARIMA(ts_data, order=(5,1,0))  # Example ARIMA parameters, can be tuned
    model_fit = model.fit()

    # Forecast next 365 days
    forecast = model_fit.forecast(steps=365)

    # Generate date range for forecasted values
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, 366)]

    # Create DataFrame for forecasted values
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

    # Plot forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[ts_column], label='Historical Data')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecasted Data', linestyle='--')
    plt.title('Forecast for Next 365 Days')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast_df
"""



def main():
    file_path = "GOOG.US_D1_cleaned.csv"
    df_all = read_and_preprocess(file_path)
    """
    # Descriptive statistics and EDA
    df_mean_3b = df_all.iloc[1545:1608]
    df_mean_3a = df_all.iloc[1609:1682]
    descriptive_statistics(df_all, df_mean_3b, df_mean_3a)
    correlation_heatmap(df_all)
    scatter_plot(df_all)
    histogram(df_all)
    line_plots(df_all)
    price_plot(df_all)

    # Model fitting and evaluation
    linear_regression(df_all)
    naive_bayes_classification(df_all)
    kmeans_clustering(df_all)
    hierarchical_clustering(df_all)
    mlp_regressor(df_all)
    pca_analysis(df_all)
    truncated_svd_analysis(df_all)
    logistic_regression_with_class_weights(df_all)
    ridge_regression(df_all)
    lasso_regression(df_all)
    """
    svm_regression(df_all)
    time_series_analysis(df_all)
    #predict_next_365_days(df_all)
if __name__ == "__main__":
    main()
