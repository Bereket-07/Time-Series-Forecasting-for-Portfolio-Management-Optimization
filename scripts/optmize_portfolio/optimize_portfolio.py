import os,logging
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_the_forecast_data(TSLA_forecast,BND_forecast,SPY_forecast):
    data = {
    "TSLA": TSLA_forecast,
    "BND": BND_forecast,
    "SPY": SPY_forecast
    }
    df_forecast = pd.DataFrame(data)
    return df_forecast
def compute_analyzed_return_and_volatility(df_forecast):
    # Daily returns
    daily_returns = df_forecast.pct_change().dropna()

    # Annualized return (assuming 252 trading days in a year)
    annualized_return = daily_returns.mean() * 252

    # Annualized covariance matrix for returns (to estimate volatility)
    annualized_cov_matrix = daily_returns.cov() * 252
    return annualized_cov_matrix , annualized_return , daily_returns
def define_portfolio_wight(annualized_return,annualized_cov_matrix):
    # Initial weights (equal allocation)
    initial_weights = np.array([1/3, 1/3, 1/3])

    # Function to calculate portfolio metrics
    # Constraints and bounds for weights
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(len(initial_weights)))

    # Optimization
    optimized_result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(annualized_return, annualized_cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    # Optimal weights
    optimal_weights = optimized_result.x
    return optimal_weights
def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate=0.02):
    _, _, sharpe_ratio = portfolio_metrics(weights, returns, cov_matrix, risk_free_rate)
    return -sharpe_ratio
def portfolio_metrics(weights, returns, cov_matrix, risk_free_rate=0.02):
    portfolio_return = np.dot(weights, returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio
def analyze_portfolio_risk(optimal_weights,annualized_return,annualized_cov_matrix,daily_returns):
    # Portfolio metrics with optimal weights
    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_metrics(
        optimal_weights, annualized_return, annualized_cov_matrix
    )

    # Value at Risk (VaR) for Tesla stock
    confidence_level = 0.95
    mean_tsla_return = daily_returns["TSLA"].mean()
    std_tsla_return = daily_returns["TSLA"].std()
    VaR_tsla = mean_tsla_return - (std_tsla_return * confidence_level)
    return portfolio_return, portfolio_volatility, sharpe_ratio
def visualiation(daily_returns,portfolio_return, portfolio_volatility):
    # Cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    for column in cumulative_returns.columns:
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=column)
    plt.title("Cumulative Returns of Forecasted Portfolio Assets")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()

    # Portfolio risk-return analysis plot
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_volatility, portfolio_return, color='red', marker='o', s=100, label='Optimal Portfolio')
    plt.title("Portfolio Risk-Return Analysis")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.show()

