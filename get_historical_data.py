from dotenv import load_dotenv
import schwabdev
import datetime
import logging
import os
from time import sleep

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwab_data.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_minute_data(client, symbol, days=90):
    """
    Get minute-by-minute data for the last specified days.
    For minute data:
    - periodType must be "day"
    - frequencyType must be "minute"
    - frequency options are: 1, 5, 10, 15, 30
    Includes extended hours and pre-market data
    """
    logger.info(f"Starting minute data fetch for symbol: {symbol}")
    all_data = []
    chunk_size = 10  # Number of days per request
    
    try:
        # Calculate end date (most recent data)
        end_date = datetime.datetime.now()
        
        # Calculate chunks working backwards from end date
        for i in range(0, days, chunk_size):
            chunk_end = end_date - datetime.timedelta(days=i)
            chunk_start = chunk_end - datetime.timedelta(days=min(chunk_size, days - i))
            
            logger.debug(f"Fetching chunk {i//chunk_size + 1} from {chunk_start.date()} to {chunk_end.date()}")
            logger.debug(f"Making API call for minute data chunk {i//chunk_size + 1} with parameters: "
                        f"symbol={symbol}, periodType=day, period={min(chunk_size, days - i)}, frequency=1")
            
            response = client.price_history(
                symbol,
                periodType="day",
                period=str(min(chunk_size, days - i)),  # Number of days for this chunk
                frequencyType="minute",
                frequency="1",  # Explicitly request 1-minute intervals
                endDate=int(chunk_end.timestamp() * 1000),  # Convert to milliseconds
                needExtendedHoursData=True  # Include extended hours data
            )
            
            logger.debug(f"API Response status code: {response.status_code}")
            response_json = response.json()
            
            if 'candles' in response_json:
                chunk_data = response_json['candles']
                logger.info(f"Retrieved {len(chunk_data)} minute candles for chunk {i//chunk_size + 1}")
                
                if chunk_data:
                    chunk_start_time = datetime.datetime.fromtimestamp(chunk_data[0]['datetime']/1000)
                    chunk_end_time = datetime.datetime.fromtimestamp(chunk_data[-1]['datetime']/1000)
                    logger.debug(f"Chunk {i//chunk_size + 1} data range: {chunk_start_time} to {chunk_end_time}")
                
                all_data.extend(chunk_data)
            else:
                logger.warning(f"No candles found in response for chunk {i//chunk_size + 1}")
                logger.debug(f"Response content: {response_json}")
            
            if i + chunk_size < days:
                logger.debug("Waiting between chunks to avoid rate limiting...")
                sleep(1)  # Add delay between chunks to avoid rate limiting
    
    except Exception as e:
        logger.error(f"Error fetching minute data for {symbol}", exc_info=True)
        logger.error(f"Exception details: {str(e)}")
    
    # Sort final data by timestamp and remove any duplicates
    all_data.sort(key=lambda x: x['datetime'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_data = []
    for item in all_data:
        if item['datetime'] not in seen:
            seen.add(item['datetime'])
            unique_data.append(item)
    
    logger.info(f"Total unique minute candles retrieved: {len(unique_data)}")
    
    if unique_data:
        start_time = datetime.datetime.fromtimestamp(unique_data[0]['datetime']/1000)
        end_time = datetime.datetime.fromtimestamp(unique_data[-1]['datetime']/1000)
        logger.info(f"Data range: {start_time} to {end_time}")
    
    return unique_data

def get_daily_data(client, symbol, days=365):
    """
    Get daily data for the last specified days
    Using year periodType for daily data
    Includes extended hours data
    """
    logger.info(f"Starting daily data fetch for symbol: {symbol}")
    
    try:
        logger.debug(f"Making API call for daily data with parameters: symbol={symbol}, periodType=year")
        response = client.price_history(
            symbol,
            periodType="year",
            period="1",  # Last 1 year
            frequency="1",  # 1 day frequency
            frequencyType="daily",
            needExtendedHoursData=True  # Include extended hours data
        )
        
        logger.debug(f"API Response status code: {response.status_code}")
        response_json = response.json()
        
        if 'candles' in response_json:
            data = response_json['candles']
            logger.info(f"Successfully retrieved {len(data)} daily candles for {symbol}")
            logger.debug(f"First candle timestamp: {data[0]['datetime'] if data else 'No data'}")
            logger.debug(f"Last candle timestamp: {data[-1]['datetime'] if data else 'No data'}")
            return data
        else:
            logger.warning(f"No candles found in response for {symbol}")
            logger.debug(f"Response content: {response_json}")
            return []
        
    except Exception as e:
        logger.error(f"Error fetching daily data for {symbol}", exc_info=True)
        logger.error(f"Exception details: {str(e)}")
        return []

def main():
    logger.info("Starting historical data collection process")
    
    # Load environment variables
    load_dotenv()
    logger.debug("Environment variables loaded")
    
    # Validate environment variables
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_APP_SECRET')
    callback_url = os.getenv('SCHWAB_CALLBACK_URL')
    
    # Log partial keys for debugging (safely)
    if app_key:
        logger.debug(f"App key found (starts with: {app_key[:4]}...)")
    if app_secret:
        logger.debug(f"App secret found (starts with: {app_secret[:4]}...)")
    if callback_url:
        logger.debug(f"Callback URL: {callback_url}")
    
    if not all([app_key, app_secret, callback_url]):
        logger.error("Missing required environment variables")
        raise Exception("Please ensure SCHWAB_APP_KEY, SCHWAB_APP_SECRET, and SCHWAB_CALLBACK_URL are set in your .env file")
    
    # Create client
    logger.info("Initializing Schwab client")
    try:
        client = schwabdev.Client(app_key, app_secret, callback_url)
        logger.info("Schwab client initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize Schwab client", exc_info=True)
        raise
    
    # Get the symbol from user input
    symbol = input("Enter the stock symbol (e.g., AAPL): ").upper()
    logger.info(f"Processing symbol: {symbol}")
    
    # Get minute data
    logger.info(f"Starting minute data collection for {symbol}")
    minute_data = get_minute_data(client, symbol)
    
    # Get daily data
    logger.info(f"Starting daily data collection for {symbol}")
    daily_data = get_daily_data(client, symbol)
    
    logger.info("Data collection process completed")

if __name__ == "__main__":
    main() 