import logging
import json
from datetime import datetime, timedelta
import time
from PyQt6.QtCore import QObject, pyqtSignal
import math
import calendar

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler
fh = logging.FileHandler('options_stream.log')
fh.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# Add handler to logger
logger.addHandler(fh)

def get_third_friday(year, month):
    """Calculate the third Friday of the given month."""
    c = calendar.monthcalendar(year, month)
    first_friday_idx = 0 if c[0][calendar.FRIDAY] else 1
    third_friday_day = c[first_friday_idx + 2][calendar.FRIDAY]
    return datetime(year, month, third_friday_day)

class SchwabOptionsHandler(QObject):
    """Handler for streaming options data from Schwab API."""
    
    data_updated = pyqtSignal(dict)
    
    def __init__(self, client=None):
        super().__init__()
        self.client = client
        self.streamer = None
        self.contracts = {}
        self.is_active = False
        logger.info("SchwabOptionsHandler initialized")
    
    def initialize(self, client):
        """Initialize with a Schwab client."""
        try:
            if not client:
                logger.error("Cannot initialize with None client")
                return False
            
            logger.info("Initializing options handler with client")
            self.client = client
            
            if not hasattr(client, 'stream'):
                logger.error("Client does not have 'stream' attribute")
                return False
            
            self.streamer = client.stream
            logger.info("Got streamer reference")
            
            if not hasattr(client, 'tokens'):
                logger.warning("Client does not have 'tokens' attribute")
            
            logger.info("Options handler successfully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing options handler: {e}", exc_info=True)
            return False
    
    def get_option_symbols(self, symbol):
        """Generate option symbols for the given underlying."""
        try:
            if not self.client:
                logger.error("Client not initialized")
                return []
            
            logger.info(f"Generating option symbols for {symbol}")
            
            # Get current price to determine strike range
            try:
                quote_response = self.client.quotes(symbol)
                if hasattr(quote_response, 'json'):
                    quote_data = quote_response.json()
                    current_price = float(quote_data.get('lastPrice', 0))
                    logger.info(f"Current price for {symbol}: ${current_price}")
                else:
                    logger.warning("Could not parse quote response")
                    current_price = 100
            except Exception as e:
                logger.error(f"Error getting quote: {e}")
                current_price = 100
            
            # Generate strikes around current price
            if current_price > 0:
                price_range = 0.20  # 20% above and below
                min_strike = max(1, round(current_price * (1 - price_range)))
                max_strike = round(current_price * (1 + price_range))
                
                # Create strike prices with proper spacing
                if current_price < 10:
                    strikes = [round(i * 0.5, 1) for i in range(int(min_strike*2), int(max_strike*2+1))]
                elif current_price < 50:
                    strikes = [round(i * 1, 0) for i in range(int(min_strike), int(max_strike+1))]
                elif current_price < 200:
                    strikes = [round(i * 5, 0) for i in range(int(min_strike/5), int(max_strike/5+1))]
                else:
                    strikes = [round(i * 10, 0) for i in range(int(min_strike/10), int(max_strike/10+1))]
            else:
                strikes = [50, 100, 150, 200, 250]
            
            # Generate expiration dates (3rd Friday for next 3 months)
            today = datetime.now()
            expiry_dates = []
            for i in range(3):
                month = (today.month + i) % 12
                if month == 0:
                    month = 12
                year = today.year + ((today.month + i - 1) // 12)
                third_friday = get_third_friday(year, month)
                expiry_str = third_friday.strftime("%y%m%d")
                expiry_dates.append(expiry_str)
            
            # Create option symbols
            option_symbols = []
            padded_symbol = symbol.ljust(6)
            
            for expiry in expiry_dates:
                for strike in strikes:
                    strike_int = int(strike)
                    strike_decimal = int((strike - strike_int) * 1000)
                    formatted_strike = f"{strike_int:05d}{strike_decimal:03d}"
                    
                    call_symbol = f"{padded_symbol}{expiry}C00{formatted_strike}"
                    put_symbol = f"{padded_symbol}{expiry}P00{formatted_strike}"
                    
                    option_symbols.append(call_symbol)
                    option_symbols.append(put_symbol)
            
            logger.info(f"Generated {len(option_symbols)} option symbols")
            return option_symbols
            
        except Exception as e:
            logger.error(f"Error generating option symbols: {e}", exc_info=True)
            return []
    
    def process_messages(self, message):
        """Process incoming messages from the streamer."""
        try:
            if isinstance(message, str):
                message_data = json.loads(message)
            else:
                message_data = message
            
            logger.debug(f"Received message: {json.dumps(message_data)[:200]}...")
            
            if "data" in message_data:
                for data_item in message_data["data"]:
                    if data_item.get("service") != "LEVELONE_OPTIONS":
                        continue
                    
                    for contract in data_item.get("content", []):
                        symbol = contract.get("key")
                        if not symbol:
                            continue
                        
                        fields = contract.get("fields", {})
                        if isinstance(fields, dict) and fields.get("1") == "Symbol not found":
                            continue
                        
                        try:
                            # Create contract data with correct field indices
                            contract_data = {
                                "symbol": symbol,
                                "bid": float(fields.get("2", 0)),
                                "ask": float(fields.get("3", 0)),
                                "last": float(fields.get("4", 0)),
                                "volume": int(float(fields.get("8", 0))),
                                "open_interest": int(float(fields.get("9", 0))),
                                "iv": float(fields.get("10", 0)) / 100.0,
                                "delta": float(fields.get("28", 0)),
                                "gamma": float(fields.get("29", 0)),
                                "theta": float(fields.get("30", 0)),
                                "vega": float(fields.get("31", 0)),
                                "rho": float(fields.get("32", 0)),
                                "strike": float(fields.get("20", 0)),
                                "contract_type": fields.get("21", "C" if "C" in symbol else "P"),
                                "underlying": fields.get("22", symbol[:6].strip()),
                                "timestamp": time.time()
                            }
                            
                            self.contracts[symbol] = contract_data
                            logger.debug(f"Updated contract data for {symbol}")
                            
                        except Exception as e:
                            logger.error(f"Error processing contract {symbol}: {e}")
                            continue
                
                if self.contracts:
                    self.data_updated.emit(self.contracts)
                    logger.debug(f"Emitted update with {len(self.contracts)} contracts")
            
            elif "response" in message_data:
                for response in message_data["response"]:
                    if response.get("service") == "LEVELONE_OPTIONS":
                        code = response.get("content", {}).get("code", -1)
                        msg = response.get("content", {}).get("msg", "Unknown")
                        logger.info(f"Options subscription response: code={code}, msg={msg}")
                        if code == 0:
                            self.is_active = True
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
    
    def start_stream(self, symbol):
        """Start streaming options data for a symbol."""
        try:
            if not self.client or not self.streamer:
                logger.error("Client or streamer not initialized")
                return False
            
            # Stop any existing stream
            self.stop_stream()
            
            # Get option symbols
            option_symbols = self.get_option_symbols(symbol)
            if not option_symbols:
                logger.error("No option symbols generated")
                return False
            
            # Start the streamer with our message handler
            self.streamer.start(self.process_messages)
            logger.info("Streamer started with message handler")
            
            # Subscribe to options data
            # Required fields for options data: 0,1,2,3,4,8,9,10,20,21,22,28,29,30,31,32
            fields = "0,1,2,3,4,8,9,10,20,21,22,28,29,30,31,32"
            batch_size = 100
            
            for i in range(0, len(option_symbols), batch_size):
                batch = option_symbols[i:i+batch_size]
                # Use the level_one_options shortcut function
                request = self.streamer.level_one_options(
                    keys=batch,
                    fields=fields,
                    command="SUBS"  # Use SUBS to overwrite previous subscriptions
                )
                self.streamer.send(request)
                logger.info(f"Sent subscription request for batch {i//batch_size + 1} with {len(batch)} symbols")
            
            self.is_active = True
            logger.info(f"Options stream started for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting options stream: {e}", exc_info=True)
            return False
    
    def stop_stream(self):
        """Stop the options stream."""
        try:
            if self.streamer:
                self.streamer.stop(clear_subscriptions=True)
                logger.info("Streamer stopped")
            
            self.is_active = False
            self.contracts.clear()
            logger.info("Options handler state reset")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream: {e}", exc_info=True)
            return False
    
    def get_contracts(self):
        """Get current contract data."""
        return self.contracts
    
    def is_stream_active(self):
        """Check if stream is active."""
        return self.is_active 