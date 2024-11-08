import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from statsmodels.tsa.seasonal import seasonal_decompose

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def data_extraction(tickers):
    logger.info("extracting the data using yfinance by tickers")
    try:
        logger.info("extracting data")
        data = yf.download(tickers,start="2018-01-01" , end = "2023-01-01")['Adj Close']
        logger.info("extraction finished succesfully")
        return data
    except Exception as e:
        logger.error(f"error occured while extracting the historical data : {e}")
def data_cleaning_and_uderstanding(df):
    logger.info("data cleaning and understanding")
    try:
        logger.info("the null values of the data")
        print(df.isna().sum())
        logger.info("if any missing value is identified they will be filled")
        df.fillna(method='ffill', inplace=True)
        logger.info("basic statistics calculating the summary statistics")
        print(df.describe())
    except Exception as e:
        logger.error(f"error occured while understaning and cleaning the dataframe:{e}")
def exploratory_data_analysis(df):
    logger.info("exploratory Data Analysis (EDA)")
    try:
        logger.info("Visualization of the Closing Price")
        df.plot(figsize=(14,7),title="Historical Closing Price")
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.show()

        logger.info("plot the Daily Percentage Change and Volatility")
        logger.info("calculating the daily returns")
        daily_returns = df.pct_change().dropna()
        logger.info("plot daily returns")
        daily_returns.plot(figsize=(14,7),title="Daily Percentage Change")
        plt.xlabel('Date')
        plt.ylabel('Daily Returns')
        plt.show()

        logger.info("Rolling Means and Standard Deviations")
        logger.info("cacluating rolling mean and std")
        rolling_mean = df.rolling(window=20).mean()
        rolling_std = df.rolling(window=20).std()

        logger.info("plot rolling mean and std")
        plt.figure(figsize=(14,7))
        plt.plot(df,label="Adjusted Close")
        plt.plot(rolling_mean , label='20-day Rolling Mean' , linestyle='--')
        plt.plot(rolling_std , label='20-day Rolling Std',linestyle='--')
        plt.title('Rolling Mean and standard deviation')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    except Exception as e:
        logger.error(f"error occured while performing exploratoty data analyiss : {e}")
def outlier_detection_and_anaylis(df):
    logger.info("outlier detection and Analysis")
    try:
        logger.info("outlier Detection")
        daily_returns = df.pct_change().dropna()
        outliers = daily_returns[(daily_returns > 3*daily_returns.std()) | (daily_returns < -3*daily_returns.std())]
        print("outliers in daily returns")
        print(outliers)
    except Exception as e:
        logger.error(f"error occured while analyzing the outliers {e}")
def seasonality_and_trends(df):
    logger.info("seasonality and trends Analysis")
    try:
        logger.info("Breaking Down the time series into trend , Seasonlaity and residuals for deeper insights")
        decomposition = seasonal_decompose(df['TSLA'] , model='additive',period=252)
        decomposition.plot()
        plt.show()
    except Exception as e:
        logger.error(f"error while performing the seasonality and trends analysis: {e}")
def risk_and_performance_metrics(df):
    logger.info("Risk and Performance Metrics")
    try:
        logger.info("calculate VaR to estimate potential losses in adverse scenarios i calcucalte aR at 95% confidence level")
        daily_returns = df.pct_change().dropna()
        VaR = daily_returns.quantile(0.05)
        print("Value ata Risk (95% confidence):")
        print(VaR)

        logger.info("calcualting the Sharoe Ratio as a measure if risk-adjsuted returns")
        mean_returns = daily_returns.mean()
        std_dev = daily_returns.std()
        # calculte annualized Sharpe Ration(assusming 2562 trading days per year)
        sharpe_ratio = (mean_returns * 252) / (std_dev * np.sqrt(252))
        print("sharpe ratio")
        print(sharpe_ratio)
    except Exception as e:
        logger.error(f"error occured while performing the Risk Perfomance Metrics {e}")


