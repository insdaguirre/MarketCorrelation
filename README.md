# S&P 500 Correlation Network Visualization

This project visualizes the correlation network of S&P 500 and Nasdaq 100 stocks using historical market data. It downloads historical prices, computes daily returns, and derives a correlation matrix from which a network graph is built. Nodes in the graph represent individual stocks, and edges exist between stocks whose correlations meet a specified threshold. The project also features basic network analysis such as degree centrality and community detection.

## Overview

- **Data Acquisition:**  
  Uses [yfinance](https://pypi.org/project/yfinance/) to pull historical adjusted close data for companies in the S&P 500, fetched directly from Wikipedia.

- **Parallel Processing:**  
  Implements multiprocessing for faster data retrieval.

- **Data Cleaning & Preprocessing:**  
  Aligns data across tickers, handles missing values, and computes daily simple returns.

- **Correlation Network Construction:**  
  Computes a correlation matrix and applies a threshold (default: 0.3) to construct a network graph using [networkx](https://networkx.github.io/).

- **Network Analysis:**  
  Performs basic analysis such as degree centrality and community detection using the greedy modularity algorithm.

- **Visualization:**  
  Visualizes the correlation network with nodes color-coded by sector and includes a legend for clarity.

## Prerequisites

Ensure you have the following installed:

- Python 3.6+
- [yfinance](https://pypi.org/project/yfinance/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [networkx](https://networkx.github.io/)
- [matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

Install the dependencies via pip:

```bash
pip install yfinance pandas numpy networkx matplotlib tqdm
