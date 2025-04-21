# Options Trading Analysis Dashboard

A real-time options trading analysis dashboard built with Python, featuring technical analysis, options chain data, and market indicators.

## Features

- Real-time stock data visualization
- Technical analysis across multiple timeframes (1min, 15min, 1h, daily)
- Options chain data with Greeks
- Advanced technical indicators:
  - Trend Following (SMA, EMA, MACD)
  - Momentum Indicators (RSI, Stochastic)
  - Volatility Indicators (Bollinger Bands, ATR)
  - Volume Analysis (OBV)
  - Pattern Recognition
  - Fair Value Gap Detection

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/options-analysis.git
cd options-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Schwab API credentials:
```
SCHWAB_APP_KEY=your_app_key
SCHWAB_APP_SECRET=your_app_secret
SCHWAB_REDIRECT_URI=your_callback_url
```

5. Run the application:
```bash
python app.py
```

## Usage

1. Enter a stock symbol in the input field
2. Click "Fetch Data" to load market data
3. Use the tabs to view different analyses:
   - Minute Data
   - Daily Data
   - Options Chain
   - Technical Analysis

## Technical Analysis Parameters

### 1-Minute Chart
- Trend Following: 30 bars
- Momentum: RSI 14 (14 min)
- Volatility: BB 20 & ATR 14
- Volume: MFI 14 (14 min)

### 15-Minute Chart
- Trend Following: 16 bars
- Momentum: RSI 14 (3½ hr)
- Volatility: BB 20 & ATR 14
- Volume: MFI 14 (3½ hr)

### 1-Hour Chart
- Trend Following: 24 bars
- Momentum: RSI 14 (14 hr)
- Volatility: BB 20 & ATR 14
- Volume: MFI 14 (14 hr)

### Daily Chart
- Trend Following: 20 bars
- Momentum: RSI 14 (14 days)
- Volatility: BB 20 & ATR 14
- Volume: MFI 14 (14 days)

## License

MIT License 