import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize

# Define the tickers
tickers = ['ALESE.PA','90N.F','GEBN.SW','ENPH','ROG.SW','NVO','688185.SS','VWS.CO','NZYM-B.CO','ECL','RYN','FSLR','TRACT.PA','BGRN','CGBIX','GRNB']

# Download the historical data
prices = yf.download(tickers, start='2017-01-01', end='2023-02-28')['Adj Close']
returns = prices.pct_change().dropna()

# Compute the covariance matrix
cov_matrix = returns.cov()

# Compute the mean returns
mean_returns = returns.mean()

# Define the number of portfolios to generate
num_portfolios = 10000

# Define the objective function to minimize
def objective_function(weights):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    portfolio_sharpe_ratio = portfolio_return / portfolio_std_dev
    return -portfolio_sharpe_ratio

# Define the constraints
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Define the bounds
bounds = tuple((0, 1) for x in range(len(tickers)))

# Generate random portfolios
results = []
for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    
    # Minimize the objective function
    optimized_weights = minimize(objective_function, weights, method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = optimized_weights.x
    
    # Compute the portfolio return and standard deviation
    portfolio_return = np.sum(mean_returns * optimized_weights)
    portfolio_std_dev = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
    
    # Store the results
    results.append((optimized_weights, portfolio_return, portfolio_std_dev))

# Convert the results into a Pandas DataFrame
columns = ['Weight', 'Returns', 'Volatility']
df_results = pd.DataFrame(results, columns=columns)

# Compute the Sharpe Ratio for each portfolio
risk_free_rate = 0.0
df_results['Sharpe Ratio'] = (df_results['Returns'] - risk_free_rate) / df_results['Volatility']

# Find the portfolio with the highest Sharpe Ratio
max_sharpe_ratio = df_results['Sharpe Ratio'].max()
best_portfolio = df_results.loc[df_results['Sharpe Ratio'] == max_sharpe_ratio]

# Plot the efficient frontier
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_results, x='Volatility', y='Returns', hue='Sharpe Ratio', palette='viridis', alpha=0.7)
plt.plot(best_portfolio['Volatility'], best_portfolio['Returns'], marker='o', markersize=10, label='Best Portfolio', color='red')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.legend()
plt.show()
