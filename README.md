# Schwab Historical Data Fetcher

A Python application for fetching historical stock data from Charles Schwab's API. This tool supports fetching both minute-by-minute and daily historical price data.

## Features

- Fetch minute-by-minute stock data (configurable for last 90 days)
- Fetch daily stock data (up to 1 year)
- Handles rate limiting and chunked requests
- Comprehensive error handling and logging
- Removes duplicate data points
- Configurable through environment variables

## Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
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
SCHWAB_CALLBACK_URL=your_callback_url
```

## Usage

Run the main script:
```bash
python get_historical_data.py
```

The script will prompt you to enter a stock symbol, and then it will fetch both minute and daily historical data for that symbol.

## Logging

The application logs detailed information to both the console and a `schwab_data.log` file. You can adjust the logging level in the code if needed.

## Error Handling

The application includes comprehensive error handling for:
- API connection issues
- Missing credentials
- Invalid responses
- Rate limiting

## Contributing

Feel free to submit issues and pull requests. 