import logging
import json
from datetime import datetime, date, timedelta
import time
import math
import calendar
import threading

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler
fh = logging.FileHandler('dash_options_stream.log')
fh.setLevel(logging.DEBUG)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(fh)
logger.addHandler(ch)

def get_third_friday(year, month):
    """
    Calculate the third Friday of the given month and year.
    Options typically expire on the third Friday of the month.
    """
    # Get the first day of the month
    c = calendar.monthcalendar(year, month)
    
    # The first Friday is the first week that contains a Friday (4)
    # If there's no Friday in the first week (index 0), use the second week (index 1)
    first_friday_idx = 0 if c[0][calendar.FRIDAY] else 1
    
    # The third Friday is two weeks after the first Friday
    third_friday_day = c[first_friday_idx + 2][calendar.FRIDAY]
    
    return datetime(year, month, third_friday_day)

class DashOptionsHandler:
    """Handler for streaming options data from Schwab."""
    
    def __init__(self, client=None):
        """Initialize the options handler."""
        self.client = client
        self.stream_active = False
        self.contracts = {}
        self.lock = threading.Lock()
        self.callbacks = []
        
    def initialize(self, client):
        """Initialize the handler with a client."""
        self.client = client
        logger.info("Options handler initialized with client")
        return True
        
    def register_callback(self, callback):
        """Register a callback function to be called when data is updated."""
        self.callbacks.append(callback)
        
    def notify_callbacks(self, data):
        """Notify all registered callbacks with the updated data."""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
                
    def _parse_contract_fields(self, contract):
        """Parse contract fields from numeric format to named fields."""
        try:
            # Skip invalid contracts
            if contract.get('1') == 'Symbol not found':
                return None
                
            # Extract symbol info from description (field 1)
            desc = contract.get('1', '')
            parts = desc.split()
            if len(parts) >= 4:
                symbol = parts[0]
                expiry = f"{parts[1]} {parts[2]}"
                strike = float(parts[3])
                contract_type = parts[4] if len(parts) > 4 else contract.get('21', '')
            else:
                # Try to parse from the key
                key = contract.get('key', '')
                if not key:
                    return None
                # Format: SPY   250516P00561000
                symbol = key[:6].strip()
                year = f"20{key[8:10]}"
                month = key[10:12]
                day = key[12:14]
                expiry = f"{month}/{day}/{year}"
                strike = float(key[16:]) / 1000
                contract_type = key[14:15]
                
            return {
                'Symbol': contract.get('key', ''),
                'Expiry': expiry,
                'Strike': float(contract.get('20', strike)),
                'Type': 'CALL' if contract_type == 'C' else 'PUT',
                'Bid': float(contract.get('2', 0)),
                'Ask': float(contract.get('3', 0)),
                'Last': float(contract.get('4', 0)),
                'Volume': int(float(contract.get('8', 0))),
                'OI': int(float(contract.get('9', 0))),
                'IV': float(contract.get('10', 0)) if float(contract.get('10', -999)) != -999 else 0,
                'Delta': float(contract.get('28', 0)) if float(contract.get('28', -999)) != -999 else 0,
                'Gamma': float(contract.get('29', 0)) if float(contract.get('29', -999)) != -999 else 0,
                'Theta': float(contract.get('30', 0)) if float(contract.get('30', -999)) != -999 else 0,
                'Vega': float(contract.get('31', 0)) if float(contract.get('31', -999)) != -999 else 0
            }
        except Exception as e:
            logger.error(f"Error parsing contract fields: {e}")
            return None
                
    def response_handler(self, message):
        """Handle responses from the Schwab API."""
        try:
            if not message or 'content' not in message:
                return
                
            content = message['content']
            if not isinstance(content, list):
                return
                
            with self.lock:
                for contract in content:
                    if isinstance(contract, dict) and 'key' in contract:
                        parsed = self._parse_contract_fields(contract)
                        if parsed:
                            self.contracts[contract['key']] = parsed
                        
            # Notify callbacks with the updated contracts
            self.notify_callbacks(list(self.contracts.values()))
            
        except Exception as e:
            logger.error(f"Error in response handler: {e}")
            
    def start_stream(self, symbol):
        """Start streaming options data for the given symbol."""
        try:
            # Get option chain first to validate symbol
            chain = self.client.option_chains(symbol)
            if not chain:
                logger.error(f"Failed to get option chain for {symbol}")
                return False
                
            # Clear existing data
            with self.lock:
                self.contracts.clear()
                
            # Start the stream
            success = self.client.levelone_option_stream(
                symbols=[symbol],
                callback=self.response_handler
            )
            
            if success:
                self.stream_active = True
                logger.info(f"Successfully started options stream for {symbol}")
            else:
                logger.error(f"Failed to start options stream for {symbol}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error starting options stream: {e}")
            return False
            
    def stop_stream(self):
        """Stop the options data stream."""
        try:
            self.client.stop_streams()
            logger.info("Stopped options stream")
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
        finally:
            self.stream_active = False
            with self.lock:
                self.contracts.clear()
                
    def is_stream_active(self):
        """Check if the stream is active."""
        return self.stream_active
        
    def get_contracts(self):
        """Get the current contracts data."""
        with self.lock:
            return list(self.contracts.values()) 