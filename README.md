# Market Correlation Network Analysis

This project analyzes the correlation network of S&P 500 stocks using historical price data. It creates a network visualization where nodes represent stocks and edges represent significant correlations between them.

## Features

- Downloads historical price data for S&P 500 stocks using yfinance
- Processes and cleans the data to ensure quality
- Calculates correlation matrix between stocks
- Creates a network visualization with:
  - Nodes colored by sector
  - Edge weights based on correlation strength
  - Interactive visualization
- Performs network analysis including:
  - Degree centrality
  - Community detection
  - Sector-based clustering

## Requirements

```bash
pip install yfinance pandas numpy networkx matplotlib tqdm
```

## Usage

```bash
python t.py
```

## Output

The script generates:
1. A network visualization of stock correlations
2. Analysis of network properties
3. Community detection results
4. Top stocks by degree centrality

## License

MIT License 