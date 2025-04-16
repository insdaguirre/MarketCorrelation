import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

# Define global start and end dates for use in the worker function.
END_DATE = datetime.datetime.now().date()
# Start date is one year ago from yesterday to ensure we have complete data
START_DATE = (END_DATE - datetime.timedelta(days=365+1))  # Use 1 year of historical data

def download_ticker(ticker):
    """
    Downloads historical adjusted close data for a single ticker.
    Returns a tuple (ticker, series) where series is a pandas Series.
    """
    try:
        # Create a Ticker object to get more reliable data
        stock = yf.Ticker(ticker)
        # Get sector information
        info = stock.info
        sector = info.get('sector', 'Unknown')
        # Get historical data with auto_adjust=True for adjusted prices
        data = stock.history(start=START_DATE, end=END_DATE, auto_adjust=True)

        # If data is empty, return None so it can be filtered out later.
        if data.empty or len(data) < 5:  # Require at least 5 days of data
            print(f"Warning: Insufficient data for {ticker}")
            return ticker, None

        # Use Close price from auto-adjusted data
        series = data['Close']
        
        # Ensure the series index is of type datetime.
        series.index = pd.to_datetime(series.index)
        
        # Verify we have enough valid data points
        if series.isna().sum() > len(series) * 0.2:  # If more than 20% is missing
            print(f"Warning: Too many missing values for {ticker}")
            return ticker, None
            
        return ticker, (series, sector)
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return ticker, None

def main():
    # -------------------------------------------------------------------------
    # 1. Define Ticker List (Nasdaq-100)
    # -------------------------------------------------------------------------
    nasdaq_100_tickers = [
        "ADBE", "AMD", "ALGN", "GOOG", "GOOGL", "AMZN", "AMGN", "AEP", "ADI", "ANSS", "AAPL",
        "AMAT", "ASML", "TEAM", "ADSK", "AZN", "AVGO", "BIDU", "BIIB", "BKNG", "CDNS", "CHTR", "CTAS",
        "CSCO", "CMCSA", "COST", "CRWD", "CSX", "CTSH", "DDOG", "DLTR", "DXCM", "EA", "EBAY", "ENPH",
        "EXC", "FAST", "FTNT", "GILD", "HON", "IDXX", "ILMN", "INTC", "INTU", "ISRG", "JD",
        "KDP", "KLAC", "KHC", "LRCX", "LCID", "LULU", "MAR", "MRVL", "MTCH", "MELI", "META", "MCHP",
        "MU", "MSFT", "NFLX", "NTES", "NVDA", "NXPI", "ORLY", "PANW", "PAYX", "PYPL", "PDD", "PEP",
        "QCOM", "REGN", "ROST", "CRM", "SIRI", "SWKS", "SBUX", "SNPS", "TMUS", "TSLA",
        "TXN", "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZTS"
    ]

    # -------------------------------------------------------------------------
    # 2. Download Price Data in Parallel
    # -------------------------------------------------------------------------
    print(f"Downloading historical data from {START_DATE} to {END_DATE}...")

    # Use 6 CPU cores for parallel downloads
    with Pool(processes=6) as pool:
        # imap returns an iterator; wrap it with tqdm for a progress bar
        results = list(tqdm(pool.imap(download_ticker, nasdaq_100_tickers),
                            total=len(nasdaq_100_tickers),
                            desc="Downloading tickers"))
    
    # Create a dictionary of ticker: series (filter out None values)
    ticker_data = {ticker: data for ticker, data in results if data is not None}
    
    if not ticker_data:
        print("No data was downloaded. Exiting.")
        return

    print(f"\nSuccessfully downloaded data for {len(ticker_data)} tickers")

    # Validate that we have actual Series objects with data
    valid_data = {}
    sectors = {}
    print("\nValidating data quality...")
    for ticker, data in ticker_data.items():
        series, sector = data
        if isinstance(series, pd.Series) and not series.empty and len(series) > 0:
            # Additional validation: check for minimum trading days and continuous data
            trading_days = len(series.dropna())
            if trading_days >= 20:  # Require at least 20 trading days
                # Check for large gaps (more than 5 consecutive NaN values)
                has_large_gaps = series.isna().astype(int).groupby(series.notna().cumsum()).sum().max() > 5
                if not has_large_gaps:
                    valid_data[ticker] = series
                    sectors[ticker] = sector
                else:
                    print(f"Skipping {ticker}: Contains large data gaps")
            else:
                print(f"Skipping {ticker}: Insufficient trading days ({trading_days})")
        else:
            print(f"Skipping {ticker}: Invalid or empty data")

    if len(valid_data) < 2:  # Need at least 2 stocks for correlation
        print(f"\nNot enough valid data for analysis. Only found {len(valid_data)} valid tickers. Need at least 2. Exiting.")
        return

    # Find the common date range across all series
    common_dates = pd.DatetimeIndex(sorted(set.intersection(*[set(s.index) for s in valid_data.values()])))
    if len(common_dates) == 0:
        print("No common dates found across tickers. Exiting.")
        return

    # Reindex all series to the common date range
    aligned_data = {ticker: series.reindex(common_dates) for ticker, series in valid_data.items()}
    
    # Combine all series into a single DataFrame aligned by date.
    df_prices = pd.DataFrame(aligned_data)
    print("Download complete.")
    
    # -------------------------------------------------------------------------
    # 3. Data Cleaning and Returns Calculation
    # -------------------------------------------------------------------------
    # Drop tickers with too many missing values (keep columns with >=80% data)
    df_prices = df_prices.dropna(axis=1, thresh=int(0.8 * len(df_prices)))
    # Fill any remaining small gaps using forward/backward fill.
    df_prices = df_prices.fillna(method='ffill').fillna(method='bfill')

    # Calculate daily simple returns
    df_returns = df_prices.pct_change().dropna(how='all')

    # -------------------------------------------------------------------------
    # 4. Compute Correlation Matrix
    # -------------------------------------------------------------------------
    corr_matrix = df_returns.corr()

    # -------------------------------------------------------------------------
    # 5. Build the Correlation Network
    # -------------------------------------------------------------------------
    threshold = 0.3  # Only keep correlations with absolute value >= threshold
    corr_thresholded = corr_matrix.mask(corr_matrix.abs() < threshold, 0)
    # Build graph: nodes = stocks; edges weighted by absolute correlation.
    G = nx.from_pandas_adjacency(corr_thresholded.abs())

    # -------------------------------------------------------------------------
    # 6. Basic Network Analysis
    # -------------------------------------------------------------------------
    print("\nNetwork Analysis Results:")
    print("------------------------")
    print(f"Number of nodes (stocks): {G.number_of_nodes()}")
    print(f"Number of edges (|corr| >= {threshold}): {G.number_of_edges()}")

    # Degree Centrality
    degree_centrality = nx.degree_centrality(G)
    degree_sorted = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 stocks by Degree Centrality:")
    for i, (ticker, centrality) in enumerate(degree_sorted[:5], start=1):
        print(f"{i}. {ticker}: {centrality:.3f}")

    # Community Detection using Greedy Modularity
    try:
        from networkx.algorithms.community import greedy_modularity_communities
    except ImportError:
        from networkx.algorithms.community.modularity_max import greedy_modularity_communities
    communities = greedy_modularity_communities(G)
    print(f"\nDetected {len(communities)} communities (Greedy Modularity):")
    for idx, com in enumerate(communities, 1):
        print(f"Community {idx} size: {len(com)}")

    # -------------------------------------------------------------------------
    # 7. Visualization
    # -------------------------------------------------------------------------
    # Create a color map for sectors
    unique_sectors = list(set(sectors.values()))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sectors)))
    sector_colors = dict(zip(unique_sectors, colors))
    
    # Create node colors list based on sectors
    node_colors = [sector_colors[sectors[node]] for node in G.nodes()]

    plt.figure(figsize=(12, 8))
    # Use a larger k value for more spread out layout and adjust seed for better visualization
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    # Draw edges first (in the background)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5)
    
    # Draw nodes with a white background for better label visibility
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors,
                          edgecolors='black', linewidths=0.5)
    
    # Add labels with a smaller font size
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=sector, markersize=10)
                      for sector, color in sector_colors.items()]
    plt.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5), title='Sectors')
    
    # Adjust layout to prevent label cutoff
    plt.margins(0.2)
    plt.subplots_adjust(right=0.85)  # Make room for the legend
    plt.title(f"Nasdaq-100 Correlation Network (Threshold = {threshold})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
