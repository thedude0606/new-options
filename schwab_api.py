import os
import logging
import schwabdev
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler
fh = logging.FileHandler('schwab_api.log')
fh.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add handler to logger
logger.addHandler(fh)

class SchwabAPI:
    """Class to handle Schwab API interactions."""
    
    def __init__(self):
        """Initialize the Schwab API client."""
        try:
            # Load environment variables
            load_dotenv()
            
            # Get credentials from environment
            app_key = os.getenv('SCHWAB_APP_KEY')
            app_secret = os.getenv('SCHWAB_APP_SECRET')
            callback_url = os.getenv('SCHWAB_REDIRECT_URI')
            
            if not app_key or not app_secret:
                raise ValueError("Missing required environment variables: SCHWAB_APP_KEY and/or SCHWAB_APP_SECRET")
            
            # Initialize client
            self.client = schwabdev.Client(
                app_key=app_key,
                app_secret=app_secret,
                callback_url=callback_url
            )
            
            # Authenticate
            if hasattr(self.client, 'authenticate'):
                self.client.authenticate()
            
            logger.info("SchwabAPI initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SchwabAPI: {e}", exc_info=True)
            raise
    
    def get_minute_data(self, symbol, days=1):
        """Get minute-by-minute data for a symbol."""
        try:
            if not symbol:
                raise ValueError("Symbol is required")
            
            logger.info(f"Fetching minute data for {symbol}")
            
            # Calculate start and end times
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get historical data
            response = self.client.get_price_history(
                symbol=symbol,
                frequency_type='minute',
                frequency=1,
                start_date=start_time.strftime('%Y-%m-%d'),
                end_date=end_time.strftime('%Y-%m-%d')
            )
            
            if not response or not hasattr(response, 'json'):
                logger.error(f"Invalid response for {symbol}")
                return None
            
            data = response.json()
            if not data or 'candles' not in data:
                logger.error(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            
            logger.info(f"Retrieved {len(df)} minute data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting minute data for {symbol}: {e}", exc_info=True)
            return None
    
    def get_daily_data(self, symbol, days=30):
        """Get daily data for a symbol."""
        try:
            if not symbol:
                raise ValueError("Symbol is required")
            
            logger.info(f"Fetching daily data for {symbol}")
            
            # Calculate start and end times
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Get historical data
            response = self.client.get_price_history(
                symbol=symbol,
                frequency_type='daily',
                frequency=1,
                start_date=start_time.strftime('%Y-%m-%d'),
                end_date=end_time.strftime('%Y-%m-%d')
            )
            
            if not response or not hasattr(response, 'json'):
                logger.error(f"Invalid response for {symbol}")
                return None
            
            data = response.json()
            if not data or 'candles' not in data:
                logger.error(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            
            logger.info(f"Retrieved {len(df)} daily data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting daily data for {symbol}: {e}", exc_info=True)
            return None
    
    def get_quote(self, symbol):
        """Get current quote for a symbol."""
        try:
            if not symbol:
                raise ValueError("Symbol is required")
            
            logger.info(f"Fetching quote for {symbol}")
            
            response = self.client.quotes(symbol)
            if not response or not hasattr(response, 'json'):
                logger.error(f"Invalid response for {symbol}")
                return None
            
            data = response.json()
            logger.info(f"Retrieved quote for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}", exc_info=True)
            return None 