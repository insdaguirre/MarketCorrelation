import yfinance as yf

tickers = ["AAPL", "VOO", "GOOG"]
data = yf.download(tickers, start = "2024-01-01", end = "2025-01-01")

print(data)