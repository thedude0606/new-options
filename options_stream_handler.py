import logging
import json
from datetime import datetime, date, timedelta
import time
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
import math
import calendar
import requests  # Import the requests library

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler
fh = logging.FileHandler('options_stream.log')
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

class OptionsStreamHandler(QObject):
    """Handler for streaming options data from Schwab."""
    
    data_updated = pyqtSignal(dict)
    
    def __init__(self, client=None):
        super().__init__()
        self.client = client
        self.streamer = None
        self.contracts = {}
        self.is_active = False
        self.field_map = {}
        self.message_queue = []  # Add message queue for processing
        self.processing_timer = QTimer()  # Add timer for processing messages
        self.processing_timer.timeout.connect(self._process_queued_messages)
        self.processing_timer.start(100)  # Process messages every 100ms
        logger.info("OptionsStreamHandler initialized")
        
    def initialize(self, client):
        """Initialize with a Schwab client."""
        try:
            if not client:
                logger.error("Cannot initialize with None client")
                return False
            
            logger.info(f"Initializing options stream handler with client")
            self.client = client
            
            # Check if the client has the required attributes
            if not hasattr(client, 'stream'):
                logger.error("Client does not have 'stream' attribute - invalid client object")
                return False
            
            # Store the streamer reference
            self.streamer = client.stream
            logger.info(f"Got streamer reference: {self.streamer}")
            
            # Get field mappings from streamer if available
            self.field_map = {}
            if hasattr(self.streamer, 'get_field_map'):
                try:
                    self.field_map = self.streamer.get_field_map('LEVELONE_OPTIONS')
                    logger.info(f"Got field map from streamer: {self.field_map}")
                except Exception as e:
                    logger.warning(f"Could not get field map from streamer: {e}")
                    # Set up default field map based on LEVELONE_OPTIONS documentation
                    self.field_map = {
                        "0": "Symbol",
                        "1": "Description",
                        "2": "Bid Price",
                        "3": "Ask Price", 
                        "4": "Last Price",
                        "5": "High Price",
                        "6": "Low Price",
                        "7": "Close Price",
                        "8": "Total Volume",
                        "9": "Open Interest",
                        "10": "Volatility",
                        "11": "Money Intrinsic Value",
                        "12": "Quote Time",
                        "13": "Trade Time",
                        "14": "Money Time Value",
                        "15": "Expiration Date",
                        "16": "Multiplier",
                        "17": "Digits",
                        "18": "Open Price",
                        "19": "Bid Size",
                        "20": "Strike Price",
                        "21": "Contract Type",
                        "22": "Underlying",
                        "23": "Ask Size",
                        "24": "Last Size",
                        "25": "Net Change",
                        "26": "Last ID",
                        "27": "Quote ID",
                        "28": "Delta",
                        "29": "Gamma",
                        "30": "Theta",
                        "31": "Vega",
                        "32": "Rho",
                        "33": "Security Status",
                        "34": "Theoretical Value",
                        "35": "Underlying Price",
                        "37": "Mark Price",
                        "38": "Quote Time in Long",
                        "39": "Trade Time in Long",
                        "40": "Exchange",
                        "41": "Exchange Name",
                        "42": "Last Trading Day",
                        "43": "Settlement Type",
                        "44": "Net Percent Change",
                        "45": "Mark Price Net Change",
                        "46": "Mark Price Percent Change",
                        "47": "Implied Yield",
                        "48": "isPennyPilot",
                        "49": "Option Root",
                        "50": "52 Week High",
                        "51": "52 Week Low",
                        "52": "Indicative Ask Price",
                        "53": "Indicative Bid Price",
                        "54": "Indicative Quote Time",
                        "55": "Exercise Type"
                    }
                    logger.info("Using default field map for LEVELONE_OPTIONS")
            
            # Based on the Stream implementation, the client should have a tokens attribute
            # that contains the access_token used for authentication
            if not hasattr(client, 'tokens'):
                logger.warning("Client does not have 'tokens' attribute - authentication may fail")
            
            # No need to create our own session - the Stream class handles the connection
            # and authentication through the websocket directly
            
            logger.info("Options stream handler successfully initialized")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing options stream handler: {e}", exc_info=True)
            return False

    def response_handler(self, message):
        """Handle incoming messages from the streamer."""
        try:
            # Always log the raw message for debugging
            logger.debug(f"Received message: {message[:1000]}")  # Log first 1000 chars to avoid huge logs
            
            if isinstance(message, str):
                try:
                    msg_data = json.loads(message)
                    logger.debug(f"Parsed JSON message: {json.dumps(msg_data, indent=2)}")
                    
                    # Handle different message types
                    if "notify" in msg_data:
                        logger.info(f"Received notification: {msg_data['notify']}")
                    elif "response" in msg_data:
                        logger.info(f"Received response: {json.dumps(msg_data['response'], indent=2)}")
                    elif "data" in msg_data:
                        self._process_messages(msg_data)
                    else:
                        logger.warning(f"Unknown message type: {list(msg_data.keys())}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON message: {e}")
            else:
                logger.warning(f"Received non-string message: {type(message)}")
        except Exception as e:
            logger.error(f"Error in response_handler: {e}", exc_info=True)

    def _process_messages(self, message_data):
        """Process incoming data messages."""
        try:
            # Add message to queue for processing
            self.message_queue.append(message_data)
            logger.debug(f"Added message to queue. Queue size: {len(self.message_queue)}")
        except Exception as e:
            logger.error(f"Error in _process_messages: {e}", exc_info=True)

    def _process_queued_messages(self):
        """Process messages from the queue."""
        try:
            if not self.message_queue:
                return

            # Process all messages in the queue
            while self.message_queue:
                message_data = self.message_queue.pop(0)
                data_items = message_data.get("data", [])
                logger.debug(f"Processing {len(data_items)} data items")
                
                contracts_updated = False
                for data_item in data_items:
                    try:
                        if data_item.get("service") != "LEVELONE_OPTIONS":
                            continue
                        
                        content = data_item.get("content", [])
                        if not content:
                            continue
                        
                        # Process each option contract
                        for contract_data in content:
                            try:
                                symbol = contract_data.get("key")
                                if not symbol:
                                    continue
                                
                                fields = contract_data.get("fields", {})
                                # If fields is not in contract_data, use the contract_data itself as fields
                                if not fields:
                                    fields = {k: v for k, v in contract_data.items() if k != "key"}
                                
                                # Skip symbols marked as not found
                                if isinstance(fields, dict) and fields.get("1") == "Symbol not found":
                                    continue
                                
                                # Parse and store contract data
                                contract_info = self._parse_contract_fields(symbol, fields)
                                if contract_info:
                                    self.contracts[symbol] = contract_info
                                    contracts_updated = True
                                    logger.debug(f"Updated contract {symbol}")
                            
                            except Exception as e:
                                logger.error(f"Error processing contract: {e}")
                                continue
                    
                    except Exception as e:
                        logger.error(f"Error processing data item: {e}")
                        continue
                
                if contracts_updated:
                    logger.info(f"Emitting update with {len(self.contracts)} contracts")
                    self.data_updated.emit(self.contracts)
                
        except Exception as e:
            logger.error(f"Error processing queued messages: {e}", exc_info=True)

    def _process_messages_with_delay(self, message):
        """Process incoming messages with a small delay to ensure proper handling."""
        try:
            # Decode JSON if needed
            if isinstance(message, str):
                try:
                    message_data = json.loads(message)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON Decode Error: {e}", exc_info=True)
                    return
            else:
                message_data = message
            
            # Handle different message types
            if "data" in message_data:
                logger.info("Processing data message")
                self._process_data_message(message_data)
            elif "response" in message_data:
                logger.info("Processing response message")
                self._process_response_message(message_data)
            elif "notify" in message_data:
                logger.info("Processing notify message")
                self._process_notify_message(message_data)
            else:
                logger.warning(f"Unknown message type: {list(message_data.keys())}")
                
        except Exception as e:
            logger.error(f"Error in process_messages: {e}", exc_info=True)

    def _process_data_message(self, message_data):
        """Process incoming data messages (e.g., quotes)."""
        data_items = message_data.get("data", [])
        logger.info(f"Processing data message with {len(data_items)} service items")
        
        # Log the entire message structure
        try:
            logger.debug(f"Full data message structure: {json.dumps(message_data, indent=2)}")
        except (TypeError, json.JSONDecodeError):
            logger.debug(f"Could not JSON serialize message: {type(message_data)}")
        
        contracts_updated = False
        for data_item in data_items:
            service = data_item.get("service")
            logger.debug(f"Processing service: {service}")
            
            if service != "LEVELONE_OPTIONS":
                logger.debug(f"Skipping non-options service: {service}")
                continue
                
            content = data_item.get("content", [])
            if not content:
                logger.warning("Empty content in options data")
                continue
                
            logger.info(f"Processing {len(content)} option contracts from service {service}")
            
            # Debug log the first contract's raw format
            if content:
                logger.info(f"Sample contract format: {json.dumps(content[0], indent=2)}")
            
            # Process each content item and update our contracts dictionary
            updated = self._process_content_items(content)
            contracts_updated = contracts_updated or updated
                
        # If any contracts were updated, emit the signal
        if contracts_updated:
            logger.info(f"Emitting update with {len(self.contracts)} contracts")
            self.data_updated.emit(self.contracts)
        else:
            logger.warning("No contracts were updated in this message")

    def _process_content_items(self, content):
        """Process a list of content items (contracts)."""
        if self.contracts is None:
            self.contracts = {}
            
        contracts_updated = False
        
        for contract in content:
            try:
                # Log the raw contract data
                logger.debug(f"Raw contract data: {json.dumps(contract, indent=2)}")
                
                # Extract the symbol (key)
                if not isinstance(contract, dict):
                    logger.warning(f"Contract is not a dictionary: {type(contract)}")
                    continue
                
                symbol = contract.get("key", "")
                if not symbol:
                    logger.warning("Contract missing symbol/key field")
                    continue
                
                # Get the fields
                fields = contract
                if "fields" in contract:
                    fields = contract["fields"]
                
                # Log the fields we're working with
                logger.debug(f"Processing fields for {symbol}: {json.dumps(fields, indent=2)}")
                
                # Parse the contract
                contract_data = self._parse_contract_fields(symbol, fields)
                if contract_data:
                    # Log before updating
                    logger.debug(f"Parsed contract data: {json.dumps(contract_data, indent=2)}")
                    
                    # Update the contract
                    self.contracts[symbol] = contract_data
                    contracts_updated = True
                    logger.info(f"Successfully processed contract: {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing contract: {e}", exc_info=True)
                continue
                
        return contracts_updated

    def _parse_contract_fields(self, symbol, fields):
        """Parse the fields dictionary for a valid contract."""
        try:
            def get_float(field_id):
                """Helper to safely get float value from field."""
                value = fields.get(str(field_id))
                if value is None:
                    return 0.0
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return 0.0

            def get_int(field_id):
                """Helper to safely get integer value from field."""
                value = fields.get(str(field_id))
                if value is None:
                    return 0
                try:
                    return int(float(value))
                except (ValueError, TypeError):
                    return 0

            # Extract key fields
            underlying = fields.get("22", "")  # Underlying
            contract_type = fields.get("21", "")  # Contract Type
            strike = get_float(20)  # Strike Price
            
            # Parse expiration date from symbol or fields
            expiry = None
            if "23" in fields and "26" in fields and "12" in fields:  # Month, Day, Year
                month = fields.get("23", "")
                day = fields.get("26", "")
                year = fields.get("12", "")
                if month and day and year:
                    try:
                        expiry = f"{year}-{month:0>2}-{day:0>2}"
                    except:
                        pass

            # Create contract data using the correct field numbers
            contract_data = {
                "symbol": symbol,
                "description": fields.get("1", ""),  # Description
                "underlying": underlying,
                "expiry": expiry,
                "contract_type": "CALL" if contract_type == "C" else "PUT",
                "strike": strike,
                "bid": get_float(2),         # Bid Price
                "ask": get_float(3),         # Ask Price
                "last": get_float(4),        # Last Price
                "high": get_float(5),        # High Price
                "low": get_float(6),         # Low Price
                "close": get_float(7),       # Close Price
                "volume": get_int(8),        # Total Volume
                "open_interest": get_int(9), # Open Interest
                "volatility": get_float(10), # Volatility
                "intrinsic_value": get_float(11),  # Money Intrinsic Value
                "multiplier": get_float(13), # Multiplier
                "digits": get_int(14),       # Digits
                "open": get_float(15),       # Open Price
                "bid_size": get_int(16),     # Bid Size
                "ask_size": get_int(17),     # Ask Size
                "last_size": get_int(18),    # Last Size
                "net_change": get_float(19), # Net Change
                "delta": get_float(28),      # Delta
                "gamma": get_float(29),      # Gamma
                "theta": get_float(30),      # Theta
                "vega": get_float(31),       # Vega
                "rho": get_float(32),        # Rho
                "security_status": fields.get("33", ""),  # Security Status
                "theoretical_value": get_float(34),  # Theoretical Option Value
                "underlying_price": get_float(35),  # Underlying Price
                "mark_price": get_float(37),  # Mark Price
                "quote_time": fields.get("38", ""),  # Quote Time in Long
                "trade_time": fields.get("39", ""),  # Trade Time in Long
                "exchange": fields.get("40", ""),  # Exchange
                "exchange_name": fields.get("41", ""),  # Exchange Name
                "last_trading_day": fields.get("42", ""),  # Last Trading Day
                "settlement_type": fields.get("43", ""),  # Settlement Type
                "net_percent_change": get_float(44),  # Net Percent Change
                "mark_price_net_change": get_float(45),  # Mark Price Net Change
                "mark_price_percent_change": get_float(46),  # Mark Price Percent Change
                "implied_yield": get_float(47),  # Implied Yield
                "is_penny_pilot": fields.get("48", ""),  # isPennyPilot
                "option_root": fields.get("49", ""),  # Option Root
                "52_week_high": get_float(50),  # 52 Week High
                "52_week_low": get_float(51),  # 52 Week Low
                "indicative_ask": get_float(52),  # Indicative Ask Price
                "indicative_bid": get_float(53),  # Indicative Bid Price
                "indicative_quote_time": fields.get("54", ""),  # Indicative Quote Time
                "exercise_type": fields.get("55", ""),  # Exercise Type
                "timestamp": time.time()
            }

            # Log successful parsing
            logger.debug(f"Successfully parsed contract: {symbol}")
            logger.debug(f"Contract data: {json.dumps(contract_data, indent=2)}")

            return contract_data

        except Exception as e:
            logger.error(f"Error parsing contract fields for {symbol}: {e}", exc_info=True)
            return None

    def _process_response_message(self, message_data):
        """Process incoming response messages (e.g., subscription status)."""
        responses = message_data.get("response", [])
        logger.info(f"Received response message with {len(responses)} items")
        print(f"RESPONSE ITEMS: {len(responses)}")
        
        for response in responses:
            service = response.get("service")
            command = response.get("command")
            content = response.get("content", {})
            
            logger.debug(f"Response for service: {service}, command: {command}")
            print(f"RESPONSE: Service={service}, Command={command}, Content={content}")
            
            if service == "LEVELONE_OPTIONS" and command == "SUBS":
                code = content.get("code", -1)
                msg = content.get("msg", "Unknown")
                logger.info(f"Options subscription response: code={code}, msg={msg}")
                print(f"OPTIONS SUBSCRIPTION: Code={code}, Message={msg}")
                if code == 0:
                    self.is_active = True
                    logger.info("Subscription successful")
                    print("SUBSCRIPTION SUCCESSFUL")
                else:
                    # Consider triggering debug if subscription fails?
                    logger.warning(f"Subscription failed: {msg}")
                    print(f"SUBSCRIPTION FAILED: {msg}")
                    # Maybe: self.debug_stream_test(underlying_symbol_from_request?)
            elif service == "ADMIN" and command == "LOGIN":
                 logger.info(f"Admin Login Response: {content}")
                 print(f"ADMIN LOGIN RESPONSE: {content}")

    def _process_notify_message(self, message_data):
        """Process incoming notification messages (e.g., heartbeats)."""
        notify = message_data.get("notify", [])
        if notify and "heartbeat" in notify[0]:
            logger.debug("Received heartbeat")
        else:
            logger.info(f"Received notification: {notify}")
            print(f"NOTIFICATION: {notify}")

    def get_option_symbols(self, symbol):
        """Generate option symbols for a given underlying symbol."""
        try:
            logger.info(f"Getting quote for {symbol} to determine strikes")
            
            # Get the quote directly using the client's methods
            try:
                # Try using client methods for quotes
                quote_response = None
                if hasattr(self.client, 'get_quotes'):
                    quote_response = self.client.get_quotes([symbol])
                elif hasattr(self.client, 'get_quotes_for_symbols'):
                    quote_response = self.client.get_quotes_for_symbols([symbol])
                elif hasattr(self.client, 'get_quote'):
                    quote_response = self.client.get_quote(symbol)
                elif hasattr(self.client, 'quotes'):
                    quote_response = self.client.quotes([symbol])
                    
                # If no method is found or they don't return usable data, make direct request
                if not quote_response or isinstance(quote_response, requests.Response):
                    logger.info("Making direct API request for quotes")
                    # Make a direct request if we have a session
                    if hasattr(self.client, 'session') and self.client.session:
                        # Use the client's session which should have authentication
                        response = self.client.session.get('https://api.schwabapi.com/marketdata/v1/quotes', 
                                                         params={'symbols': symbol, 'indicative': 'False'})
                        if response.status_code == 200:
                            # Extract JSON data from response
                            quote_response = response.json()
                            logger.debug(f"Direct API quote response (JSON): {quote_response}")
                        else:
                            logger.error(f"Direct API request failed: {response.status_code} - {response.text}")
                            raise Exception(f"API request failed: {response.status_code}")
                    else:
                        # If the client.get_quotes returned a Response object, try to extract JSON
                        if isinstance(quote_response, requests.Response):
                            if quote_response.status_code == 200:
                                quote_response = quote_response.json()
                                logger.debug(f"Extracted JSON from response: {quote_response}")
                            else:
                                logger.error(f"Quote request failed: {quote_response.status_code} - {quote_response.text}")
                                raise Exception(f"Quote request failed: {quote_response.status_code}")
                        else:
                            logger.error("No session available and no usable quote data")
                            raise Exception("No session available for direct API request")
            
            except Exception as e:
                logger.error(f"Error calling quote method: {e}", exc_info=True)
                # Don't fall back to default price - we want real data
                raise e
            
            # Based on logs, let's inspect the actual structure of the response
            logger.debug(f"Quote response type: {type(quote_response)}, Value: {quote_response}")
            
            # Extract current price based on Schwab API response format
            current_price = None
            
            # Check response type and handle accordingly
            if isinstance(quote_response, dict):
                # Schwab API response format often has a 'quotes' list
                if 'quotes' in quote_response and isinstance(quote_response['quotes'], list):
                    quotes = quote_response['quotes']
                    if quotes and len(quotes) > 0:
                        quote_data = quotes[0]  # First quote in the list
                        logger.debug(f"Found quote data: {quote_data}")
                        
                        # Try different possible field names
                        if 'lastPrice' in quote_data:
                            current_price = quote_data['lastPrice']
                        elif 'regularMarketLastPrice' in quote_data:
                            current_price = quote_data['regularMarketLastPrice']
                        elif 'price' in quote_data:
                            current_price = quote_data['price']
                        # Try more fields that might contain the price
                        elif 'bidPrice' in quote_data:
                            current_price = quote_data['bidPrice']
                        elif 'askPrice' in quote_data:
                            current_price = quote_data['askPrice']
                        # Even more possibilities
                        elif 'mark' in quote_data:
                            current_price = quote_data['mark']
                        elif 'marketValue' in quote_data:
                            current_price = quote_data['marketValue']
                
                # Format where the symbol is a key
                elif symbol in quote_response:
                    quote_data = quote_response[symbol]
                    if isinstance(quote_data, dict):
                        logger.debug(f"Found quote data for symbol {symbol}: {quote_data}")
                        
                        if 'lastPrice' in quote_data:
                            current_price = quote_data['lastPrice']
                        elif 'regularMarketLastPrice' in quote_data:
                            current_price = quote_data['regularMarketLastPrice']
                        elif 'quote' in quote_data and isinstance(quote_data['quote'], dict):
                            if 'lastPrice' in quote_data['quote']:
                                current_price = quote_data['quote']['lastPrice']
                        elif 'regular' in quote_data and isinstance(quote_data['regular'], dict):
                            if 'regularMarketLastPrice' in quote_data['regular']:
                                current_price = quote_data['regular']['regularMarketLastPrice']
                
                # Direct response format
                elif 'lastPrice' in quote_response:
                    current_price = quote_response['lastPrice']
                elif 'regularMarketLastPrice' in quote_response:
                    current_price = quote_response['regularMarketLastPrice']
                elif 'price' in quote_response:
                    current_price = quote_response['price']
                elif 'mark' in quote_response:
                    current_price = quote_response['mark']
            
            # Log the extracted price or error
            if current_price is not None:
                logger.info(f"Successfully extracted price for {symbol}: ${current_price}")
            else:
                # If we still don't have a price, try a different approach
                logger.warning("Could not extract price from standard response format")
                
                # Check if we have a direct simple value in the response
                if isinstance(quote_response, (int, float)):
                    current_price = float(quote_response)
                    logger.info(f"Using direct numeric response as price: ${current_price}")
                # Last fallback - use client method to get current price
                elif hasattr(self.client, 'get_price') and callable(getattr(self.client, 'get_price')):
                    current_price = self.client.get_price(symbol)
                    logger.info(f"Got price using get_price method: ${current_price}")
                else:
                    # Log the response for debugging
                    logger.error(f"Could not extract price from response: {quote_response}")
                    raise Exception("Could not extract price from quote response")
            
            logger.info(f"Current price for {symbol}: ${current_price}")
            return self._generate_option_symbols(symbol, current_price)
            
        except Exception as e:
            logger.error(f"Error generating option symbols: {e}", exc_info=True)
            raise e  # Re-raise the exception to prevent falling back to fake data

    def _generate_option_symbols(self, symbol, current_price):
        """Helper to generate option symbols based on a symbol and its price."""
        try:
            # Generate strikes around current price (±20%)
            min_strike = math.floor(current_price * 0.8)
            max_strike = math.ceil(current_price * 1.2)
            strike_step = 5  # $5 increments
            
            strikes = range(min_strike, max_strike + strike_step, strike_step)
            logger.info(f"Generated {len(strikes)} strikes from ${min_strike} to ${max_strike}")
            
            # Get next 3 expiration dates
            today = date.today()
            expirations = []
            
            # Add today's expiration if it's a trading day (0DTE)
            if self.is_trading_day(today):
                expirations.append(datetime.combine(today, datetime.min.time()))
                logger.info(f"Added 0DTE expiration for {today}")
            
            # Add next 3 monthly expirations
            for i in range(3):
                next_month = today.replace(day=1) + timedelta(days=32*i)
                next_month = next_month.replace(day=1)
                third_friday = get_third_friday(next_month.year, next_month.month)
                if third_friday.date() > today:
                    expirations.append(third_friday)
            
            # Generate option symbols
            option_symbols = []
            for expiry in expirations:
                # Format expiry as YYMMDD (6 characters)
                expiry_str = expiry.strftime("%y%m%d")
                
                for strike in strikes:
                    # Format strike with leading zeros to 8 digits (5+3=8 characters)
                    # First 5 digits are the whole number part, last 3 are decimal
                    strike_str = f"{int(strike):05d}000"
                    
                    # Format underlying symbol to 6 characters (including spaces)
                    padded_symbol = symbol.ljust(6)
                    
                    # Generate both calls and puts
                    # Format: [Underlying Symbol (6 chars) | Expiration (6 chars) | Call/Put (1 char) | Strike Price (8 chars)]
                    call_symbol = f"{padded_symbol}{expiry_str}C{strike_str}"
                    put_symbol = f"{padded_symbol}{expiry_str}P{strike_str}"
                    
                    option_symbols.extend([call_symbol, put_symbol])
            
            logger.info(f"Generated {len(option_symbols)} option symbols for {symbol}")
            if option_symbols:
                logger.debug(f"Sample option symbols: {option_symbols[:6]}")
            
            return option_symbols
            
        except Exception as e:
            logger.error(f"Error in _generate_option_symbols: {e}", exc_info=True)
            return []

    def is_trading_day(self, date_obj):
        """Check if the given date is a trading day (Monday-Friday)."""
        return date_obj.weekday() < 5  # 0-4 are Monday-Friday

    def discover_valid_option_symbols(self, candidate_symbols, batch_size=20, wait_time=2):
        """
        Probe the streamer with batches of candidate symbols, collect those that do NOT return 'Symbol not found'.
        Returns a list of valid symbols.
        """
        logger.info(f"Starting discovery mode for {len(candidate_symbols)} candidate symbols...")
        print(f"DISCOVERY MODE: Probing {len(candidate_symbols)} candidate symbols...")
        valid_symbols = []
        temp_contracts = {}
        # Use a minimal handler to capture responses
        def discovery_handler(message):
            try:
                if isinstance(message, str):
                    message_data = json.loads(message)
                else:
                    message_data = message
                if "data" in message_data:
                    for data_item in message_data["data"]:
                        if data_item.get("service") != "LEVELONE_OPTIONS":
                            continue
                        for contract in data_item.get("content", []):
                            symbol = contract.get("key")
                            fields = contract.get("fields", {})
                            if symbol and not (isinstance(fields, dict) and fields.get("1") == "Symbol not found"):
                                temp_contracts[symbol] = fields
            except Exception as e:
                logger.error(f"Error in discovery handler: {e}")
        # Stop and start streamer with discovery handler
        try:
            try:
                self.streamer.stop(clear_subscriptions=True)
            except Exception:
                pass
            self.streamer.start(discovery_handler)
            fields = "0,1"  # Only need Symbol and Description
            for i in range(0, len(candidate_symbols), batch_size):
                batch = candidate_symbols[i:i+batch_size]
                logger.info(f"Discovery: Subscribing to batch {i//batch_size + 1} with {len(batch)} symbols")
                request = self.streamer.level_one_options(
                    keys=','.join(batch),
                    fields=fields,
                    command="SUBS"
                )
                self.streamer.send(request)
                time.sleep(wait_time)  # Wait for responses
            # Collect valid symbols
            valid_symbols = list(temp_contracts.keys())
            logger.info(f"Discovery complete: {len(valid_symbols)} valid symbols found.")
            print(f"DISCOVERY COMPLETE: {len(valid_symbols)} valid symbols found.")
        except Exception as e:
            logger.error(f"Error during discovery mode: {e}", exc_info=True)
        finally:
            # Always restart streamer with main handler
            try:
                self.streamer.stop(clear_subscriptions=True)
            except Exception:
                pass
            self.streamer.start(self.response_handler)
        return valid_symbols

    def start_stream(self, symbol):
        """Start streaming options data for a symbol."""
        try:
            if not self.client:
                logger.error("Client not initialized")
                return False
            
            # Clear existing contracts when starting a new stream
            self.contracts = {}
            
            # Make sure we have a streamer reference
            if not self.streamer and hasattr(self.client, 'stream'):
                self.streamer = self.client.stream
                logger.info("Retrieved streamer reference from client")
            
            if not self.streamer:
                logger.error("No streamer available")
                return False
            
            # Stop any existing stream
            try:
                if hasattr(self.streamer, 'stop'):
                    self.streamer.stop(clear_subscriptions=True)
                    logger.info("Stopped existing stream")
            except Exception as e:
                logger.warning(f"Error stopping existing stream: {e}")
            
            # Start the streamer with our response handler
            if hasattr(self.streamer, 'start'):
                self.streamer.start(self.response_handler)
                logger.info("Started streamer with response_handler")
            else:
                logger.error("Streamer doesn't have a start method")
                return False
            
            # Get option symbols for the underlying
            try:
                option_symbols = self.get_option_symbols(symbol)
                if not option_symbols:
                    logger.error(f"No option symbols generated for {symbol}")
                    return False
            except Exception as e:
                logger.error(f"Failed to get option symbols: {e}")
                return False
            
            # Log some sample symbols to verify format
            if option_symbols:
                logger.info(f"Generated {len(option_symbols)} option symbols. First few: {option_symbols[:5]}")
            
            # Subscribe to options data in batches of 500 (maximum allowed)
            batch_size = 500
            # Request all fields from 0 to 55
            fields = ','.join(str(i) for i in range(56))
            
            for i in range(0, len(option_symbols), batch_size):
                batch = option_symbols[i:i + batch_size]
                logger.info(f"Subscribing to batch {i//batch_size + 1} of {len(option_symbols)//batch_size + 1} with {len(batch)} options")
                
                try:
                    # Use the streamer's level_one_options method
                    if hasattr(self.streamer, 'level_one_options'):
                        # Join symbols with commas for the keys parameter
                        keys = ','.join(batch)
                        
                        # Use SUBS command to overwrite existing subscriptions for this batch
                        request = self.streamer.level_one_options(
                            keys=keys,
                            fields=fields,
                            command="SUBS"  # Use SUBS to overwrite existing subscriptions
                        )
                        
                        # Send the request
                        logger.info(f"Sending subscription request for batch {i//batch_size + 1}")
                        self.streamer.send(request)
                        logger.info(f"Subscription request sent for batch {i//batch_size + 1}")
                    else:
                        # Fallback to basic request if level_one_options is not available
                        subscription = {
                            "service": "LEVELONE_OPTIONS",
                            "command": "SUBS",  # Use SUBS to overwrite existing subscriptions
                            "parameters": {
                                "keys": ','.join(batch),
                                "fields": fields
                            }
                        }
                        logger.info(f"Sending basic subscription request for batch {i//batch_size + 1}")
                        self.streamer.send(subscription)
                        logger.info(f"Basic subscription request sent for batch {i//batch_size + 1}")
                        
                except Exception as e:
                    logger.error(f"Error sending subscription for batch {i//batch_size + 1}: {e}", exc_info=True)
                    return False
            
            # Set stream as active
            self.is_active = True
            logger.info(f"Successfully subscribed to options data for {symbol}")
            
            # Create an initial data update signal even if no data yet
            self.data_updated.emit(self.contracts)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting options stream: {e}", exc_info=True)
            return False

    def _fetch_option_chain(self, symbol):
        """Fetch option chain directly from the API."""
        try:
            logger.info(f"Fetching option chain for {symbol} from API")
            
            # Check for session access
            session = None
            if hasattr(self.client, 'session') and self.client.session:
                session = self.client.session
            elif hasattr(self.client, 'http') and self.client.http:
                session = self.client.http
            elif hasattr(self.client, 'api') and hasattr(self.client.api, 'session'):
                session = self.client.api.session
            
            if not session:
                logger.warning("No session available for API request, generating option symbols manually")
                # Fall back to a hardcoded price to generate option symbols
                current_price = 0
                # Try to get price using a different method
                try:
                    if hasattr(self.client, 'get_price') and callable(getattr(self.client, 'get_price')):
                        current_price = self.client.get_price(symbol)
                        logger.info(f"Got price using get_price method: ${current_price}")
                    else:
                        # Use hardcoded prices as last resort
                        default_prices = {"SPY": 500.0, "QQQ": 400.0, "AAPL": 170.0, "MSFT": 400.0, "GOOGL": 170.0, "AMZN": 180.0, "TSLA": 180.0}
                        current_price = default_prices.get(symbol, 100.0)
                        logger.info(f"Using default price for {symbol}: ${current_price}")
                except Exception as e:
                    logger.error(f"Error getting price: {e}")
                    # Use a reasonable default price
                    current_price = 100.0
                
                # Generate option symbols using the price
                return self._generate_option_symbols(symbol, current_price)
            
            # If we have a session, try to get the option chain
            try:
                # Get current date
                today = date.today()
                
                # Calculate from date (today) and to date (90 days in future)
                from_date = today.strftime("%Y-%m-%d")
                to_date = (today + timedelta(days=90)).strftime("%Y-%m-%d")
                
                # Try different API endpoints that might be available
                # First try the options chain endpoint
                try:
                    response = session.get(
                        'https://api.schwabapi.com/trader/v1/option/chains',
                        params={
                            'symbol': symbol,
                            'fromDate': from_date,
                            'toDate': to_date,
                            'range': 'ALL',  # Get all available strikes
                            'expMonth': 'ALL'  # Get all expiration months
                        }
                    )
                    
                    if response.status_code == 200:
                        # Parse the response
                        chain_data = response.json()
                        logger.debug(f"Option chain response: {chain_data}")
                        
                        # Extract option symbols
                        option_symbols = self._extract_option_symbols_from_chain(chain_data)
                        if option_symbols:
                            return option_symbols
                    else:
                        logger.warning(f"Options chain API request failed: {response.status_code} - {response.text}")
                except Exception as e:
                    logger.warning(f"Error with options chain endpoint: {e}")
                
                # Try the option expiration dates endpoint
                try:
                    response = session.get(
                        'https://api.schwabapi.com/trader/v1/option/expirationdates',
                        params={'symbol': symbol}
                    )
                    
                    if response.status_code == 200:
                        expiration_data = response.json()
                        logger.debug(f"Option expiration data: {expiration_data}")
                        
                        # Extract expirations and generate option symbols
                        return self._generate_option_symbols_from_expirations(symbol, expiration_data)
                    else:
                        logger.warning(f"Expiration dates API request failed: {response.status_code} - {response.text}")
                except Exception as e:
                    logger.warning(f"Error with expiration dates endpoint: {e}")
                
                # Fall back to manual generation as last resort
                logger.warning("API requests failed, generating option symbols manually")
                # Try to get current price
                try:
                    quote_response = session.get(
                        'https://api.schwabapi.com/marketdata/v1/quotes',
                        params={'symbols': symbol}
                    )
                    
                    if quote_response.status_code == 200:
                        quote_data = quote_response.json()
                        current_price = self._extract_price_from_quote(quote_data, symbol)
                        if current_price:
                            return self._generate_option_symbols(symbol, current_price)
                except Exception as e:
                    logger.warning(f"Error getting price for manual generation: {e}")
                
                # Use default price as absolute last resort
                logger.warning("Using default price for manual generation")
                default_price = 100.0
                return self._generate_option_symbols(symbol, default_price)
                
            except Exception as e:
                logger.error(f"Error fetching option chain with session: {e}", exc_info=True)
                # Fall back to manual generation
                return self._generate_option_symbols(symbol, 100.0)
            
        except Exception as e:
            logger.error(f"Error in _fetch_option_chain: {e}", exc_info=True)
            return []
    
    def _extract_option_symbols_from_chain(self, chain_data):
        """Extract option symbols from chain data."""
        option_symbols = []
        
        try:
            # Different possible formats of the chain data
            if 'callExpDateMap' in chain_data and 'putExpDateMap' in chain_data:
                # TDA-style format
                for exp_map in [chain_data['callExpDateMap'], chain_data['putExpDateMap']]:
                    for date_str, strikes in exp_map.items():
                        for strike_str, options in strikes.items():
                            for option in options:
                                if 'symbol' in option:
                                    option_symbols.append(option['symbol'])
            elif 'options' in chain_data:
                # Direct list of options
                for option in chain_data['options']:
                    if 'symbol' in option:
                        option_symbols.append(option['symbol'])
            elif 'optionChain' in chain_data:
                # Another possible format
                chain = chain_data['optionChain']
                if isinstance(chain, list):
                    for item in chain:
                        if 'option' in item:
                            for option in item['option']:
                                if 'symbol' in option:
                                    option_symbols.append(option['symbol'])
            
            logger.info(f"Extracted {len(option_symbols)} option symbols from chain data")
            return option_symbols
        except Exception as e:
            logger.error(f"Error extracting option symbols from chain: {e}")
            return []
    
    def _generate_option_symbols_from_expirations(self, symbol, expiration_data):
        """Generate option symbols using expiration data."""
        option_symbols = []
        
        try:
            # Extract expirations from the data
            expirations = []
            if 'expirationDates' in expiration_data:
                expirations = expiration_data['expirationDates']
            elif isinstance(expiration_data, list):
                expirations = expiration_data
            
            if not expirations:
                logger.warning("No expirations found in data")
                return []
            
            # Get current price
            current_price = 100.0  # Default
            if hasattr(self.client, 'get_price'):
                try:
                    current_price = self.client.get_price(symbol)
                except Exception:
                    pass
            
            # Generate strikes based on current price
            min_strike = math.floor(current_price * 0.8)
            max_strike = math.ceil(current_price * 1.2)
            strike_step = 5  # $5 increments
            
            strikes = range(min_strike, max_strike + strike_step, strike_step)
            
            # Generate symbols for each expiration and strike
            for exp_date in expirations[:3]:  # Use first 3 expirations
                # Format the expiration date - depends on the format in the response
                if isinstance(exp_date, str):
                    # Try to parse the date string
                    try:
                        # Format might be "YYYY-MM-DD"
                        exp_datetime = datetime.strptime(exp_date, "%Y-%m-%d")
                        expiry_str = exp_datetime.strftime("%y%m%d")
                    except ValueError:
                        try:
                            # Or might be "MM/DD/YYYY"
                            exp_datetime = datetime.strptime(exp_date, "%m/%d/%Y")
                            expiry_str = exp_datetime.strftime("%y%m%d")
                        except ValueError:
                            # Use as is if we can't parse
                            expiry_str = exp_date
                else:
                    # If it's already a datetime
                    expiry_str = exp_date.strftime("%y%m%d") if isinstance(exp_date, datetime) else str(exp_date)
                
                # Generate symbols for each strike
                for strike in strikes:
                    # Format strike with leading zeros to 8 digits
                    strike_str = f"{strike:08d}"
                    # Generate both calls and puts
                    call_symbol = f"{symbol:<6}{expiry_str}C{strike_str}"
                    put_symbol = f"{symbol:<6}{expiry_str}P{strike_str}"
                    option_symbols.extend([call_symbol, put_symbol])
            
            logger.info(f"Generated {len(option_symbols)} option symbols from expirations")
            return option_symbols
            
        except Exception as e:
            logger.error(f"Error generating symbols from expirations: {e}")
            return []
    
    def _extract_price_from_quote(self, quote_data, symbol):
        """Extract price from quote data."""
        try:
            if 'quotes' in quote_data and isinstance(quote_data['quotes'], list):
                for quote in quote_data['quotes']:
                    if 'lastPrice' in quote:
                        return quote['lastPrice']
                    elif 'price' in quote:
                        return quote['price']
                    elif 'bidPrice' in quote:
                        return quote['bidPrice']
            
            # Try other formats
            if symbol in quote_data:
                if 'lastPrice' in quote_data[symbol]:
                    return quote_data[symbol]['lastPrice']
            
            # Direct format
            if 'lastPrice' in quote_data:
                return quote_data['lastPrice']
            
            return None
        except Exception as e:
            logger.error(f"Error extracting price from quote: {e}")
            return None

    def debug_stream_test(self, symbol):
        """Create a minimal test stream for debugging purposes."""
        try:
            logger.info("Starting minimal debug stream test")
            print("STARTING MINIMAL DEBUG STREAM TEST")
            
            if not self.client or not self.streamer:
                logger.error("Client or streamer not initialized for debug test")
                return False
            
            # Always restart the streamer
            try:
                self.streamer.stop(clear_subscriptions=True)
                logger.info("Stopped existing streamer")
            except Exception as e:
                logger.debug(f"No active streamer to stop: {e}")
            
            # Start the streamer with our handler
            self.streamer.start(self.response_handler)
            
            # Create a hardcoded test symbol in the right format (padded to 6 chars, etc.)
            padded_symbol = symbol.ljust(6)
            
            # Create one each of a call and put option with different strikes and expiries
            # Use current month + 1 and next year for expiry
            today = datetime.now()
            month = (today.month + 1) % 12
            if month == 0:
                month = 12
            year = today.year + 1
            
            # Format the expiry date
            expiry = f"{str(year)[-2:]}{month:02d}20"  # YY-MM-20 (20th day)
            
            # Create sample symbols with varying strikes
            test_symbols = []
            for strike in [100, 200, 300, 400, 500]:
                # Format the strike with leading zeros (8 digits)
                formatted_strike = f"{strike:08d}"
                
                # Create call and put symbols
                call_symbol = f"{padded_symbol}{expiry}C00{formatted_strike}"
                put_symbol = f"{padded_symbol}{expiry}P00{formatted_strike}"
                
                test_symbols.append(call_symbol)
                test_symbols.append(put_symbol)
            
            logger.info(f"Using test symbols: {test_symbols}")
            print(f"USING TEST SYMBOLS: {test_symbols}")
            
            # Subscribe to the test symbols with all fields
            fields = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22"
            
            # Subscribe to the test symbols
            request = self.streamer.level_one_options(
                keys=','.join(test_symbols),
                fields=fields,
                command="SUBS"
            )
            
            response = self.streamer.send(request)
            logger.info(f"Debug test subscription response: {response}")
            print(f"DEBUG TEST SUBSCRIPTION RESPONSE: {response}")
            
            # Always consider this active for testing
            self.is_active = True
            
            # Create dummy contracts for UI display
            for symbol in test_symbols:
                is_call = "C" in symbol
                strike = float(symbol[-8:])
                
                self.contracts[symbol] = {
                    "symbol": symbol,
                    "bid": 1.25 if is_call else 0.75,
                    "ask": 1.50 if is_call else 1.00,
                    "last": 1.35 if is_call else 0.85,
                    "volume": 1000,
                    "open_interest": 5000,
                    "iv": 0.35,
                    "delta": 0.65 if is_call else -0.45,
                    "gamma": 0.05,
                    "theta": -0.04,
                    "vega": 0.06,
                    "rho": 0.01,
                    "strike": strike,
                    "contract_type": "C" if is_call else "P",
                    "underlying": symbol[:6].strip(),
                    "timestamp": time.time()
                }
            
            # Emit an update with our dummy data
            self.data_updated.emit(self.contracts)
            logger.info(f"Created {len(self.contracts)} dummy contracts for testing")
            print(f"CREATED {len(self.contracts)} DUMMY CONTRACTS FOR TESTING")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in debug stream test: {e}", exc_info=True)
            print(f"ERROR IN DEBUG STREAM TEST: {e}")
            return False

    def stop_stream(self):
        """Stop the options stream."""
        try:
            logger.info("Stopping options stream...")
            
            if self.streamer and hasattr(self.streamer, 'stop'):
                # Use the Stream class's stop method which will properly log out
                # and clear subscriptions
                self.streamer.stop(clear_subscriptions=True)
                logger.info("Streamer stopped and subscriptions cleared")
            else:
                logger.warning("Streamer not available or missing stop method")
            
            # Reset our state regardless of streamer status
            self.is_active = False
            self.contracts.clear()
            logger.info("Stream handler state reset")
            return True
        except Exception as e:
            logger.error(f"Error stopping stream: {str(e)}", exc_info=True)
            return False
    
    def get_contracts(self):
        """Get current contract data."""
        return self.contracts
    
    def is_stream_active(self):
        """Check if stream is active."""
        return self.is_active

    def test_stream(self, symbol="SPY"):
        """Test function to examine stream field mappings and data format."""
        try:
            logger.info("=== Starting Stream Test ===")
            
            if not self.client or not self.streamer:
                logger.error("Client or streamer not initialized")
                return
            
            # Stop any existing stream
            try:
                self.streamer.stop(clear_subscriptions=True)
                logger.info("Stopped existing stream")
            except Exception as e:
                logger.warning(f"Error stopping existing stream: {e}")
            
            # Create a test handler to examine raw messages
            def test_handler(message):
                try:
                    logger.info("\n=== Test Handler Message ===")
                    if isinstance(message, str):
                        msg_data = json.loads(message)
                        logger.info(f"Message type: {type(msg_data)}")
                        logger.info(f"Message keys: {list(msg_data.keys())}")
                        logger.info(f"Raw message:\n{json.dumps(msg_data, indent=2)}")
                        
                        # Check for field definitions
                        if "response" in msg_data:
                            for resp in msg_data["response"]:
                                if resp.get("service") == "LEVELONE_OPTIONS":
                                    logger.info("\n=== Field Definitions ===")
                                    logger.info(json.dumps(resp, indent=2))
                        
                        # Check for actual data
                        if "data" in msg_data:
                            for data_item in msg_data["data"]:
                                if data_item.get("service") == "LEVELONE_OPTIONS":
                                    logger.info("\n=== Options Data Sample ===")
                                    content = data_item.get("content", [])
                                    if content:
                                        logger.info(f"First contract data:\n{json.dumps(content[0], indent=2)}")
                    else:
                        logger.info(f"Non-string message: {type(message)}")
                except Exception as e:
                    logger.error(f"Error in test handler: {e}", exc_info=True)
            
            # Start streamer with test handler
            self.streamer.start(test_handler)
            logger.info("Started streamer with test handler")
            
            # Create a test option symbol
            base_symbol = f"{symbol:<6}"  # Pad to 6 chars
            today = datetime.now()
            # Use next month for expiry
            next_month = today.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)
            expiry = next_month.strftime("%y%m%d")
            # Create one call and one put near current price
            test_symbols = [
                f"{base_symbol}{expiry}C00500000",  # $500 call
                f"{base_symbol}{expiry}P00500000"   # $500 put
            ]
            
            logger.info(f"Test symbols: {test_symbols}")
            
            # Subscribe to all fields
            all_fields = list(range(0, 50))  # Request a wide range of fields
            fields_str = ",".join(str(f) for f in all_fields)
            
            # Send subscription request
            if hasattr(self.streamer, 'level_one_options'):
                request = self.streamer.level_one_options(
                    keys=",".join(test_symbols),
                    fields=fields_str,
                    command="SUBS"
                )
                logger.info(f"Subscription request: {json.dumps(request, indent=2)}")
                
                response = self.streamer.send(request)
                logger.info(f"Subscription response: {response}")
            
            logger.info("Test subscription sent. Check the logs for incoming data.")
            logger.info("=== End Stream Test Setup ===")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in test_stream: {e}", exc_info=True)
            return False

# No longer needed - we create the instance in app.py
# stream_handler = OptionsStreamHandler()
logger.info("Options stream handler module initialized") 