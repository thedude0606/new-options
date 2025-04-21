from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import pandas as pd
from datetime import datetime
import os
from get_historical_data import get_minute_data, get_daily_data
from dotenv import load_dotenv
import schwabdev
import logging
from dash.exceptions import PreventUpdate
import dash
import time
from datetime import timezone
import pytz
from options_stream_handler import OptionsStreamHandler
import json
import sys
import plotly.graph_objs as go
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QTableWidget, QTableWidgetItem, QTabWidget, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from schwab_api import SchwabAPI
from schwab_options_handler import SchwabOptionsHandler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import random
from technical_indicators import TechnicalAnalysis
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('app')

# Initialize Dash app
logger.info("Initializing Dash application")
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Stock Data Dashboard"

# Update environment variable loading
load_dotenv()
# Use the app_key and app_secret variables directly as in the examples
app_key = os.getenv('app_key')
app_secret = os.getenv('app_secret')
callback_url = os.getenv('callback_url')

# Fall back to SCHWAB_ prefixed variables if not found
if not app_key or not app_secret:
    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_APP_SECRET')
    callback_url = os.getenv('SCHWAB_REDIRECT_URI')

logger.info(f"Initializing client with app_key: {app_key[:5]}... and app_secret: {app_secret[:5]}...")
client = schwabdev.Client(app_key=app_key, app_secret=app_secret, callback_url=callback_url)

# Initialize options handler with the client
class Signal:
    """Simple signal/slot implementation."""
    def __init__(self):
        self._callbacks = []
        
    def connect(self, callback):
        self._callbacks.append(callback)
        
    def emit(self, *args, **kwargs):
        for callback in self._callbacks:
            callback(*args, **kwargs)
            
    def disconnect(self, callback):
        if callback in self._callbacks:
            self._callbacks.remove(callback)

class OptionsStreamHandler:
    def __init__(self):
        """Initialize the options handler."""
        self.client = None
        self.stream_active = False
        self.contracts = {}
        self.lock = threading.Lock()
        self.data_updated = Signal()
        self.debug_mode = True
        logger.info("OptionsStreamHandler initialized")
        
    def initialize(self, client):
        """Initialize with a client."""
        self.client = client
        logger.info("OptionsStreamHandler initialized with client")
        return True

    def on_message(self, message):
        """Handle incoming messages from the stream."""
        try:
            # Log the complete message structure
            logger.info("=== Received Message ===")
            logger.info(f"Message Type: {type(message)}")
            logger.info(f"Message Content: {json.dumps(message, indent=2) if isinstance(message, (dict, list)) else message}")
            
            if not message:
                logger.warning("Received empty message")
                return
            
            # Parse the message if it's a string
            if isinstance(message, str):
                try:
                    message = json.loads(message)
                    logger.debug("Successfully parsed message string to JSON")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message as JSON: {str(e)}")
                    return
            
            # Process the message based on its structure
            if isinstance(message, dict):
                if 'data' in message:
                    logger.debug("Processing message with 'data' field")
                    for data_item in message['data']:
                        if data_item.get('service') == 'LEVELONE_OPTIONS':
                            logger.debug("Found LEVELONE_OPTIONS data")
                            content = data_item.get('content', [])
                            logger.debug(f"Content items: {len(content)}")
                            for item in content:
                                if 'key' in item:
                                    symbol = item['key']
                                    logger.debug(f"Processing option: {symbol}")
                                    # Update the contract data
                                    with self.lock:
                                        if symbol not in self.contracts:
                                            self.contracts[symbol] = {}
                                        # Update fields
                                        for field_id, value in item.items():
                                            if field_id != 'key':
                                                self.contracts[symbol][field_id] = value
                                        logger.debug(f"Updated contract data for {symbol}")
                            
                    # Emit update with all contracts
                    with self.lock:
                        contracts_list = list(self.contracts.values())
                        logger.info(f"Emitting update with {len(contracts_list)} contracts")
                        self.data_updated.emit(contracts_list)
                else:
                    logger.warning("Message does not contain 'data' field")
            else:
                logger.warning(f"Unexpected message type: {type(message)}")
            
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}", exc_info=True)
            
    def start_stream(self, symbol):
        """Start streaming options data for the given symbol."""
        try:
            # Get option chain first to validate symbol and get option symbols
            logger.info(f"Fetching option chain for {symbol}")
            chain = self.client.option_chains(
                symbol=symbol,
                contractType='ALL',
                range='ALL',
                strikeCount=10  # Limit for testing
            )
            
            if not chain:
                logger.error("No response from options chain API")
                return False
                
            # Convert to JSON if needed
            chain_data = chain.json() if hasattr(chain, 'json') else chain
            logger.debug(f"Option chain response: {json.dumps(chain_data, indent=2)}")
            
            # Extract option symbols from chain
            option_symbols = []
            
            # Process calls and puts
            for exp_map_key in ['callExpDateMap', 'putExpDateMap']:
                if exp_map_key in chain_data:
                    for exp_date, strikes in chain_data[exp_map_key].items():
                        for strike, options in strikes.items():
                            for option in options:
                                option_symbols.append(option['symbol'])
                                logger.info(f"Found option: {option['symbol']}")
            
            if not option_symbols:
                logger.error("No option symbols found in chain")
                return False
                
            logger.info(f"Found {len(option_symbols)} options to stream")
            
            # Create subscription request with all fields
            fields = [str(i) for i in range(56)]  # Fields 0-55
            
            subscription_request = {
                "requests": [{
                    "service": "LEVELONE_OPTIONS",
                    "command": "SUBS",
                    "requestid": "1",
                    "parameters": {
                        "keys": ','.join(option_symbols),
                        "fields": ','.join(fields)
                    }
                }]
            }
            
            logger.info("Sending subscription request:")
            logger.info(json.dumps(subscription_request, indent=2))
            
            # Start the stream
            try:
                if not hasattr(self.client, 'stream'):
                    logger.error("Client does not have 'stream' attribute")
                    return False
                
                # Clear existing data
                with self.lock:
                    self.contracts.clear()
                    logger.debug("Cleared existing contracts")
                
                # Start the streamer with our message handler
                self.client.stream.start(self.on_message)
                logger.info("Streamer started with message handler")
                
                # Subscribe to options data
                request = self.client.stream.level_one_options(
                    keys=','.join(option_symbols),
                    fields=','.join(fields),
                    command="SUBS"
                )
                self.client.stream.send(request)
                logger.info("Sent subscription request")
                
                with self.lock:
                    self.stream_active = True
                    logger.info(f"Successfully started options stream for {len(option_symbols)} contracts")
                
                return True
                
            except Exception as e:
                logger.error(f"Error starting stream: {e}", exc_info=True)
                return False
            
        except Exception as e:
            logger.error(f"Error starting options stream: {str(e)}", exc_info=True)
            return False
    
    def stop_stream(self):
        """Stop the options data stream."""
        try:
            if hasattr(self.client, 'stream'):
                self.client.stream.stop()
                logger.info("Stopped options stream")
        except Exception as e:
            logger.error(f"Error stopping stream: {str(e)}", exc_info=True)
        finally:
            with self.lock:
                self.stream_active = False
                self.contracts.clear()
                logger.debug("Cleared contracts and set stream inactive")
    
    def is_stream_active(self):
        """Check if the stream is active."""
        with self.lock:
            return self.stream_active
    
    def get_contracts(self):
        """Get the current contracts data."""
        with self.lock:
            contracts = list(self.contracts.values())
            logger.debug(f"Returning {len(contracts)} contracts")
            if contracts:
                logger.debug(f"Sample contract: {json.dumps(contracts[0], indent=2)}")
            return contracts

# Create and initialize the options handler
options_handler = OptionsStreamHandler()
options_handler.initialize(client)
logger.info("Options handler created and initialized")

# Create a global variable to store the latest options data
latest_options_data = {}

# Add debug logging to the signal handler
def handle_options_data_update(data):
    global latest_options_data
    logger.debug("=== Options Data Update Signal Received ===")
    logger.debug(f"Data type: {type(data)}")
    logger.debug(f"Data content: {json.dumps(data, indent=2)}")
    
    # Make a deep copy to avoid reference issues
    latest_options_data = data.copy() if data else []
    logger.debug(f"Updated latest_options_data with {len(data)} contracts")
    
    if data:
        # Log sample data for verification
        sample_contracts = data[:3] if len(data) > 3 else data
        logger.debug(f"Sample contracts: {json.dumps(sample_contracts, indent=2)}")
        for contract in sample_contracts:
            logger.debug(f"Contract details: {json.dumps(contract, indent=2)}")

# Connect the signal to our handler function
options_handler.data_updated.connect(handle_options_data_update)
logger.info("Connected to options data update signal")

# Add these lines near the top after the imports
stored_minute_data = []
stored_daily_data = []

def initialize_client():
    """Initialize the Schwab client."""
    global client, options_handler, latest_options_data
    try:
        app_key = os.getenv('SCHWAB_APP_KEY')
        app_secret = os.getenv('SCHWAB_APP_SECRET')
        if not app_key or not app_secret:
            raise ValueError("Missing required environment variables")
        
        # Create a new client instance
        client = schwabdev.Client(app_key=app_key, app_secret=app_secret)
        
        # Reset the options handler with the new client
        if options_handler is not None:
            options_handler.initialize(client)
            
        # Clear existing data
        latest_options_data = {}
        
        return client
    except Exception as e:
        logger.error(f"Error initializing client: {str(e)}")
        raise

def process_data(data):
    """Process data into table format."""
    if not data:
        return []
    
    rows = []
    for row in data:
        timestamp = row.get('datetime', 0)
        et_tz = pytz.timezone('US/Eastern')
        dt = datetime.fromtimestamp(timestamp/1000, tz=pytz.UTC)
        dt_et = dt.astimezone(et_tz)
        formatted_time = dt_et.strftime('%Y-%m-%d %H:%M:%S ET')
        
        rows.append({
            'Time': formatted_time,
            'Open': row.get('open', ''),
            'High': row.get('high', ''),
            'Low': row.get('low', ''),
            'Close': row.get('close', ''),
            'Volume': row.get('volume', '')
        })
    return rows

# Add field mapping constants at the top of the file
OPTION_FIELD_MAP = {
    "0": "symbol",
    "1": "description",
    "2": "bid_price",
    "3": "ask_price",
    "4": "last_price",
    "5": "high_price",
    "6": "low_price",
    "7": "close_price",
    "8": "total_volume",
    "9": "open_interest",
    "10": "volatility",
    "11": "quote_time",
    "12": "trade_time",
    "13": "money_intrinsic_value",
    "14": "quote_day",
    "15": "trade_day",
    "16": "expiration_year",
    "17": "multiplier",
    "18": "digits",
    "19": "open_price",
    "20": "bid_size",
    "21": "ask_size",
    "22": "last_size",
    "23": "net_change",
    "24": "strike_price",
    "25": "contract_type",
    "26": "underlying",
    "27": "expiration_month",
    "28": "deliverables",
    "29": "time_value",
    "30": "expiration_day",
    "31": "days_to_expiration",
    "32": "delta",
    "33": "gamma",
    "34": "theta",
    "35": "vega",
    "36": "rho",
    "37": "security_status",
    "38": "theoretical_option_value",
    "39": "underlying_price",
    "40": "uv_expiration_type",
    "41": "mark",
    "42": "percent_change"
}

def format_price(price):
    """Format price value"""
    if price is None:
        return None
    try:
        return round(float(price), 2)
    except (ValueError, TypeError):
        return None

def format_percent(percent):
    """Format percent value"""
    if percent is None:
        return None
    try:
        return round(float(percent), 2)
    except (ValueError, TypeError):
        return None

def format_volume(volume):
    """Format volume value"""
    if volume is None:
        return None
    try:
        vol = int(float(volume))
        if vol >= 1000000:
            return f"{vol/1000000:.2f}M"
        elif vol >= 1000:
            return f"{vol/1000:.2f}K"
        return str(vol)
    except (ValueError, TypeError):
        return None

def format_expiration(year, month, day):
    """Format expiration date"""
    if None in (year, month, day):
        return None
    try:
        return f"{year}-{month:02d}-{day:02d}"
    except (ValueError, TypeError):
        return f"{year}-{month}-{day}"

def process_option_data(data):
    """Process raw option data into formatted display data"""
    try:
        if not data:
            return None
            
        logger.debug(f"Raw data: {json.dumps(data, indent=2)}")
        
        # Extract fields using correct field numbers from Schwab API
        processed = {}
        
        # Handle the key/symbol field
        processed["symbol"] = data.get("key") if "key" in data else data.get("0", "")
        
        # Extract numeric fields with proper type conversion and error handling
        numeric_fields = {
            "bid": ("2", float, 0),  # Bid Price
            "ask": ("3", float, 0),  # Ask Price
            "last": ("4", float, 0),  # Last Price
            "high": ("5", float, 0),  # High Price
            "low": ("6", float, 0),  # Low Price
            "close": ("7", float, 0),  # Close Price
            "volume": ("8", int, 0),  # Total Volume
            "open_interest": ("9", int, 0),  # Open Interest
            "volatility": ("10", float, 0),  # Volatility
            "money_intrinsic": ("11", float, 0),  # Money Intrinsic Value
            "strike": ("20", float, 0),  # Strike Price
            "delta": ("28", float, 0),  # Delta
            "gamma": ("29", float, 0),  # Gamma
            "theta": ("30", float, 0),  # Theta
            "vega": ("31", float, 0),  # Vega
            "rho": ("32", float, 0),  # Rho
            "time_value": ("25", float, 0),  # Time Value
            "theoretical_value": ("34", float, 0),  # Theoretical Option Value
            "underlying_price": ("35", float, 0),  # Underlying Price
            "mark_price": ("37", float, 0)  # Mark Price
        }
        
        for field, (field_id, convert_func, default) in numeric_fields.items():
            try:
                value = data.get(field_id)
                if value is not None and value != "":
                    processed[field] = convert_func(float(value))
                else:
                    processed[field] = default
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting {field} (field {field_id}): {str(e)}")
                processed[field] = default
        
        # Extract string fields
        string_fields = {
            "description": ("1", ""),  # Description
            "contract_type": ("21", ""),  # Contract Type
            "underlying": ("22", ""),  # Underlying
            "deliverables": ("24", ""),  # Deliverables
            "exchange": ("40", ""),  # Exchange
            "security_status": ("33", "")  # Security Status
        }
        
        for field, (field_id, default) in string_fields.items():
            processed[field] = data.get(field_id, default)
        
        # Extract date components
        processed["expiration_year"] = int(data.get("12", 0))
        processed["expiration_month"] = int(data.get("23", 0))
        processed["expiration_day"] = int(data.get("26", 0))
        
        # Set option type
        processed["type"] = "CALL" if processed["contract_type"] == "C" else "PUT"
        
        # Format expiration date
        if all(processed.get(x, 0) > 0 for x in ["expiration_year", "expiration_month", "expiration_day"]):
            processed["expiration"] = f"{processed['expiration_year']}-{processed['expiration_month']:02d}-{processed['expiration_day']:02d}"
        else:
            processed["expiration"] = "Unknown"
        
        # Calculate bid-ask spread
        bid_ask_spread = processed['ask'] - processed['bid'] if processed['ask'] > 0 and processed['bid'] > 0 else 0
        
        # Format the data for display
        formatted = {
            "type": processed["type"],
            "strike": f"${processed['strike']:.2f}",
            "expiration": processed["expiration"],
            "bid": f"${processed['bid']:.2f}",
            "ask": f"${processed['ask']:.2f}",
            "last": f"${processed['last']:.2f}",
            "volume": f"{processed['volume']:,}",
            "open_interest": f"{processed['open_interest']:,}",
            "bid_ask_spread": f"${bid_ask_spread:.2f}",
            "volatility": f"{processed['volatility']:.1f}%",
            "delta": f"{processed['delta']:.3f}",
            "gamma": f"{processed['gamma']:.3f}",
            "theta": f"${processed['theta']:.2f}",
            "in_the_money": "Yes" if processed["money_intrinsic"] > 0 else "No"
        }
        
        logger.debug(f"Processed option data: {json.dumps(formatted, indent=2)}")
        return formatted
        
    except Exception as e:
        logger.error(f"Error processing option data: {str(e)}", exc_info=True)
        return None

# App layout
app.layout = html.Div([
    dcc.Store(id='options-data-store', data={'calls': [], 'puts': []}),
    dcc.Store(id='technical-analysis-store', data={}),
    dcc.Interval(id='interval-component', interval=5000),  # 5 second interval
    html.H1("Stock Data Dashboard", style={'textAlign': 'center'}),
    
    # Single input section for all data
    html.Div([
        dcc.Input(
            id='symbol-input',
            type='text',
            placeholder='Enter stock symbol',
            style={'margin': '10px'}
        ),
        html.Button(
            'Fetch Data',
            id='fetch-button',
            n_clicks=0,
            style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}
        ),
        html.Button(
            'Start Stream',
            id='start-stream-btn',
            n_clicks=0,
            style={'margin': '10px', 'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}
        ),
        html.Button(
            'Stop Stream',
            id='stop-stream-btn',
            n_clicks=0,
            style={'margin': '10px', 'backgroundColor': '#f44336', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}
        ),
        html.Div(id='status-message', style={'margin': '10px', 'color': '#666'}),
        html.Div(id='stream-status', style={'margin': '10px', 'color': '#666'})
    ], style={'textAlign': 'center'}),

    # Debug information section
    html.Div([
        html.H4("Debug Information", style={'margin': '10px', 'color': '#666'}),
        html.Div(id='debug-info', style={
            'margin': '10px',
            'padding': '10px',
            'backgroundColor': '#f5f5f5',
            'borderRadius': '5px',
            'fontFamily': 'monospace',
            'whiteSpace': 'pre-wrap'
        })
    ], style={'margin': '20px'}),

    # Tabs for different data views
    dcc.Tabs([
        # Minute Data Tab
        dcc.Tab(label='Minute Data', children=[
            html.Div([
                html.H3("Minute Data"),
                dash_table.DataTable(
                    id='minute-data-table',
                    columns=[
                        {'name': 'Time', 'id': 'Time'},
                        {'name': 'Open', 'id': 'Open'},
                        {'name': 'High', 'id': 'High'},
                        {'name': 'Low', 'id': 'Low'},
                        {'name': 'Close', 'id': 'Close'},
                        {'name': 'Volume', 'id': 'Volume'}
                    ],
                    style_table={'height': '400px', 'overflowY': 'auto'}
                )
            ])
        ]),
        
        # Daily Data Tab
        dcc.Tab(label='Daily Data', children=[
            html.Div([
                html.H3("Daily Data"),
                dash_table.DataTable(
                    id='daily-data-table',
                    columns=[
                        {'name': 'Time', 'id': 'Time'},
                        {'name': 'Open', 'id': 'Open'},
                        {'name': 'High', 'id': 'High'},
                        {'name': 'Low', 'id': 'Low'},
                        {'name': 'Close', 'id': 'Close'},
                        {'name': 'Volume', 'id': 'Volume'}
                    ],
                    style_table={'height': '400px', 'overflowY': 'auto'}
                )
            ])
        ]),
        
        # Options Chain Tab
        dcc.Tab(label='Options Chain', children=[
            html.Div([
                html.H2("Options Chain", style={'textAlign': 'center', 'marginBottom': '20px'}),
                
                # Calls Table
                html.Div([
                    html.H3("Calls", style={'textAlign': 'center', 'color': '#0066cc'}),
                    dash_table.DataTable(
                        id='calls-table',
                        columns=[
                            {'name': 'Strike', 'id': 'strike', 'type': 'numeric'},
                            {'name': 'Expiration', 'id': 'expiration'},
                            {'name': 'Bid', 'id': 'bid'},
                            {'name': 'Ask', 'id': 'ask'},
                            {'name': 'Last', 'id': 'last'},
                            {'name': 'Volume', 'id': 'volume'},
                            {'name': 'OI', 'id': 'open_interest'},
                            {'name': 'Spread', 'id': 'bid_ask_spread'},
                            {'name': 'IV', 'id': 'volatility'},
                            {'name': 'Delta', 'id': 'delta'},
                            {'name': 'Gamma', 'id': 'gamma'},
                            {'name': 'Theta', 'id': 'theta'},
                            {'name': 'ITM', 'id': 'in_the_money'}
                        ],
                        filter_action='native',
                        sort_action='native',
                        sort_mode='multi',
                        style_table={
                            'height': '400px',
                            'overflowY': 'auto'
                        },
                        style_cell={
                            'textAlign': 'right',
                            'padding': '10px',
                            'fontFamily': 'monospace'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'in_the_money', 'filter_query': '{in_the_money} = "Yes"'},
                                'backgroundColor': '#e6ffe6'
                            }
                        ],
                        page_size=25
                    )
                ], style={'margin': '20px'}),
                
                # Puts Table
                html.Div([
                    html.H3("Puts", style={'textAlign': 'center', 'color': '#cc0000'}),
                    dash_table.DataTable(
                        id='puts-table',
                        columns=[
                            {'name': 'Strike', 'id': 'strike', 'type': 'numeric'},
                            {'name': 'Expiration', 'id': 'expiration'},
                            {'name': 'Bid', 'id': 'bid'},
                            {'name': 'Ask', 'id': 'ask'},
                            {'name': 'Last', 'id': 'last'},
                            {'name': 'Volume', 'id': 'volume'},
                            {'name': 'OI', 'id': 'open_interest'},
                            {'name': 'Spread', 'id': 'bid_ask_spread'},
                            {'name': 'IV', 'id': 'volatility'},
                            {'name': 'Delta', 'id': 'delta'},
                            {'name': 'Gamma', 'id': 'gamma'},
                            {'name': 'Theta', 'id': 'theta'},
                            {'name': 'ITM', 'id': 'in_the_money'}
                        ],
                        filter_action='native',
                        sort_action='native',
                        sort_mode='multi',
                        style_table={
                            'height': '400px',
                            'overflowY': 'auto'
                        },
                        style_cell={
                            'textAlign': 'right',
                            'padding': '10px',
                            'fontFamily': 'monospace'
                        },
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': 'bold',
                            'textAlign': 'center'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'in_the_money', 'filter_query': '{in_the_money} = "Yes"'},
                                'backgroundColor': '#e6ffe6'
                            }
                        ],
                        page_size=25
                    )
                ], style={'margin': '20px'})
            ])
        ]),
        
        # Technical Analysis Tab
        dcc.Tab(label='Technical Analysis', children=[
            html.Div([
                html.H3("Technical Analysis", style={'textAlign': 'center'}),
                html.Div(id='technical-analysis-output', style={
                    'padding': '20px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '5px',
                    'margin': '20px'
                })
            ])
        ])
    ]),
    
    dcc.Interval(id='options-update-interval', interval=1000),
])

@callback(
    [Output('minute-data-table', 'data'),
     Output('daily-data-table', 'data'),
     Output('status-message', 'children')],
    [Input('fetch-button', 'n_clicks')],
    [State('symbol-input', 'value')]
)
def update_data(n_clicks, symbol):
    global stored_minute_data, stored_daily_data
    if not symbol:
        raise PreventUpdate
    symbol = symbol.upper()
    try:
        client = initialize_client()
        stored_minute_data = get_minute_data(client, symbol)
        stored_daily_data = get_daily_data(client, symbol)
        minute_processed = process_data(stored_minute_data)
        daily_processed = process_data(stored_daily_data)
        status_message = f"Data fetched for {symbol}"
        return minute_processed, daily_processed, status_message
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [], [], error_msg

@callback(
    [Output('options-data-store', 'data'),
     Output('stream-status', 'children'),
     Output('debug-info', 'children')],
    [Input('start-stream-btn', 'n_clicks'),
     Input('stop-stream-btn', 'n_clicks'),
     Input('options-update-interval', 'n_intervals')],
    [State('symbol-input', 'value'),
     State('options-data-store', 'data')],
    prevent_initial_call=True
)
def update_stream_data(start_clicks, stop_clicks, n_intervals, symbol, current_data):
    logger.debug("=== Stream Data Update Callback ===")
    
    try:
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        logger.info(f"Callback triggered by: {trigger_id}")
        
        if trigger_id == 'start-stream-btn' and symbol:
            logger.info(f"Starting stream for symbol: {symbol}")
            success = options_handler.start_stream(symbol)
            if success:
                return current_data, f"Stream started for {symbol}", f"Stream started for {symbol}"
            else:
                return current_data, "Failed to start stream", "Error starting stream"
        
        elif trigger_id == 'stop-stream-btn':
            logger.info("Stopping stream")
            options_handler.stop_stream()
            return current_data, "Stream stopped", "Stream stopped"
        
        elif trigger_id == 'options-update-interval':
            if not options_handler.is_stream_active():
                logger.debug("Stream is not active")
                return current_data, "Stream inactive", "Stream is not active"
            
            try:
                contracts = options_handler.get_contracts()
                logger.debug(f"Retrieved {len(contracts)} contracts from handler")
                
                if not contracts:
                    logger.debug("No contracts received")
                    return current_data, "Waiting for data...", "No contracts received yet"
                
                # Process contracts into calls and puts
                calls = []
                puts = []
                
                for contract in contracts:
                    try:
                        logger.debug(f"Processing contract: {json.dumps(contract, indent=2)}")
                        processed = process_option_data(contract)
                        if processed:
                            logger.debug(f"Formatted contract: {json.dumps(processed, indent=2)}")
                            if processed["type"] == "CALL":
                                calls.append(processed)
                            else:
                                puts.append(processed)
                        else:
                            logger.warning(f"Failed to process contract: {json.dumps(contract, indent=2)}")
                    except Exception as e:
                        logger.error(f"Error processing contract: {str(e)}", exc_info=True)
                        continue
                
                new_data = {'calls': calls, 'puts': puts}
                status = f"Streaming data - {len(calls)} calls, {len(puts)} puts"
                debug_info = f"Last update: {datetime.now().strftime('%H:%M:%S')}\nProcessed {len(contracts)} contracts\nCalls: {len(calls)}\nPuts: {len(puts)}"
                
                logger.info(f"Updated data: {len(calls)} calls, {len(puts)} puts")
                if calls or puts:
                    logger.debug(f"Sample call: {json.dumps(calls[0] if calls else puts[0], indent=2)}")
                
                return new_data, status, debug_info
                
            except Exception as e:
                logger.error(f"Error during update: {str(e)}", exc_info=True)
                return current_data, f"Error: {str(e)}", f"Error during update: {str(e)}"
        
        return current_data, "No action taken", "No action taken"
        
    except Exception as e:
        logger.error(f"Callback error: {str(e)}", exc_info=True)
        return current_data, f"Error: {str(e)}", f"Callback error: {str(e)}"

@callback(
    [Output('calls-table', 'data'),
     Output('puts-table', 'data')],
    [Input('options-data-store', 'data')]
)
def update_options_tables(data):
    if not data or not isinstance(data, dict):
        return [], []
    
    try:
        calls = []
        puts = []
        
        # Process calls and puts separately
        for call in data.get('calls', []):
            calls.append(call)
        
        for put in data.get('puts', []):
            puts.append(put)
        
        # Sort by expiration date and ITM proximity
        def sort_key(option):
            try:
                if option['expiration'] == "Unknown":
                    exp_date = datetime.max
                else:
                    exp_date = datetime.strptime(option['expiration'], '%Y-%m-%d')
                
                # Calculate ITM proximity (absolute difference from current price)
                strike = float(option['strike'].replace('$', ''))
                underlying_price = float(data.get('underlying_price', 0))
                itm_proximity = abs(strike - underlying_price)
                
                return (exp_date, itm_proximity)
            except (ValueError, KeyError):
                return (datetime.max, float('inf'))
        
        # Sort both lists
        calls.sort(key=sort_key)
        puts.sort(key=sort_key)
        
        return calls, puts
        
    except Exception as e:
        logger.error(f"Error in update_options_tables: {str(e)}", exc_info=True)
        return [], []

@callback(
    Output('technical-analysis-output', 'children'),
    [Input('symbol-input', 'value'),
     Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_technical_analysis(symbol, n):
    """Update technical analysis indicators."""
    logger.info(f"Technical analysis callback triggered for symbol: {symbol}")
    
    if not symbol:
        logger.warning("No symbol provided")
        return html.Div("Please enter a symbol")
    
    try:
        logger.info(f"Fetching technical analysis for {symbol}")
        
        # Get data from Schwab API
        client = initialize_client()
        logger.debug("Client initialized")
        
        daily_data = get_daily_data(client, symbol)
        logger.debug(f"Retrieved {len(daily_data) if daily_data else 0} data points from Schwab API")
        
        if not daily_data:
            logger.error(f"No data found for symbol {symbol}")
            return html.Div(f"No data found for symbol {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(daily_data)
        logger.debug(f"Created DataFrame with columns: {df.columns.tolist()}")
        
        df = df.rename(columns={
            'datetime': 'Time',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        logger.debug(f"Renamed columns to: {df.columns.tolist()}")
        
        # Convert timestamp to datetime
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        
        # Sort by time
        df = df.sort_values('Time')
        
        logger.debug(f"DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame head:\n{df.head()}")
        
        # Initialize technical analysis
        logger.info("Initializing TechnicalAnalysis class")
        ta = TechnicalAnalysis(df)
        
        # Calculate all timeframes
        logger.info("Calculating multi-timeframe analysis")
        analysis = ta.calculate_multi_timeframe_analysis()
        
        # Create layout for indicators
        logger.info("Creating indicator display layout")
        
        # Helper function to format indicator value
        def format_value(value, is_percent=False):
            if value is None:
                return "N/A"
            
            # Handle dictionary values
            if isinstance(value, dict):
                value = value.get('value', value)
            
            # Handle numpy types
            if hasattr(value, 'item'):
                value = value.item()
                
            if is_percent:
                return f"{value:.2f}%"
            return f"{value:.2f}"
        
        # Helper function to create indicator section
        def create_indicator_section(title, indicators):
            return html.Div([
                html.H5(title),
                html.Div([
                    html.Div([
                        html.Strong(f"{k}: "),
                        html.Span(format_value(v)),
                        html.Span(f" ({v.get('signal', 'N/A') if isinstance(v, dict) else 'N/A'})")
                    ]) for k, v in indicators.items()
                ])
            ])
        
        # Create sections for each timeframe
        timeframe_sections = []
        for timeframe, data in analysis.items():
            sections = [
                html.H4(f"{timeframe} Timeframe Analysis"),
                
                # Trend Indicators
                create_indicator_section("Trend Indicators", {
                    "SMA(20)": data['trend_indicators']['sma']['sma20'],
                    "SMA(50)": data['trend_indicators']['sma']['sma50'],
                    "SMA(200)": data['trend_indicators']['sma']['sma200'],
                    "EMA(12)": data['trend_indicators']['ema']['ema12'],
                    "EMA(26)": data['trend_indicators']['ema']['ema26'],
                    "MACD": data['trend_indicators']['macd'],
                    "HMA": data['trend_indicators']['hma'],
                    "VWAP": data['trend_indicators']['vwap']
                }),
                
                # Ichimoku Cloud
                html.Div([
                    html.H5("Ichimoku Cloud"),
                    html.Div([
                        html.Div([
                            html.Strong("Tenkan: "),
                            html.Span(format_value(data['trend_indicators']['ichimoku']['tenkan'])),
                            html.Span(f" ({'Above' if data['trend_indicators']['ichimoku']['tenkan_above_kijun'] else 'Below'} Kijun)")
                        ]),
                        html.Div([
                            html.Strong("Kijun: "),
                            html.Span(format_value(data['trend_indicators']['ichimoku']['kijun']))
                        ]),
                        html.Div([
                            html.Strong("Senkou A: "),
                            html.Span(format_value(data['trend_indicators']['ichimoku']['senkou_a']))
                        ]),
                        html.Div([
                            html.Strong("Senkou B: "),
                            html.Span(format_value(data['trend_indicators']['ichimoku']['senkou_b']))
                        ]),
                        html.Div([
                            html.Strong("Cloud Color: "),
                            html.Span(data['trend_indicators']['ichimoku']['cloud_color'].title()),
                            html.Span(f" ({data['trend_indicators']['ichimoku']['signal']})")
                        ])
                    ])
                ]),
                
                # Momentum Indicators
                create_indicator_section("Momentum Indicators", {
                    "RSI": data['momentum_indicators']['rsi'],
                    "Stochastic K": data['momentum_indicators']['stochastic']['k'],
                    "Stochastic D": data['momentum_indicators']['stochastic']['d'],
                    "CCI": data['momentum_indicators']['cci'],
                    "ROC": data['momentum_indicators']['roc'],
                    "ADX": data['momentum_indicators']['adx']
                }),
                
                # Volatility Indicators
                create_indicator_section("Volatility Indicators", {
                    "ATR": data['volatility_indicators']['atr'],
                    "Bollinger Upper": data['volatility_indicators']['bollinger_bands']['upper'],
                    "Bollinger Middle": data['volatility_indicators']['bollinger_bands']['middle'],
                    "Bollinger Lower": data['volatility_indicators']['bollinger_bands']['lower'],
                    "StdDev": data['volatility_indicators']['stddev'],
                    "Keltner Upper": data['volatility_indicators']['keltner_channels']['upper'],
                    "Keltner Middle": data['volatility_indicators']['keltner_channels']['middle'],
                    "Keltner Lower": data['volatility_indicators']['keltner_channels']['lower'],
                    "Donchian Upper": data['volatility_indicators']['donchian_channel']['upper'],
                    "Donchian Lower": data['volatility_indicators']['donchian_channel']['lower']
                }),
                
                # Volume Indicators
                create_indicator_section("Volume Indicators", {
                    "OBV": data['volume_indicators']['obv'],
                    "AD": data['volume_indicators']['ad'],
                    "MFI": data['volume_indicators']['mfi']
                }),
                
                # Candlestick Patterns
                html.Div([
                    html.H5("Candlestick Patterns"),
                    html.Div([
                        html.Div([
                            html.Strong("Single Candle: "),
                            html.Span(", ".join(data['candlestick_patterns']['single_candle'].keys()) if data['candlestick_patterns']['single_candle'] else "None")
                        ]),
                        html.Div([
                            html.Strong("Two Candle: "),
                            html.Span(", ".join(data['candlestick_patterns']['two_candle'].keys()) if data['candlestick_patterns']['two_candle'] else "None")
                        ]),
                        html.Div([
                            html.Strong("Three Candle: "),
                            html.Span(", ".join(data['candlestick_patterns']['three_candle'].keys()) if data['candlestick_patterns']['three_candle'] else "None")
                        ])
                    ])
                ]),
                
                # Fair Value Gaps
                html.Div([
                    html.H5("Fair Value Gaps"),
                    html.Div([
                        html.Div([
                            html.Strong("Count: "),
                            html.Span(str(data['fair_value_gaps']['count']))
                        ]),
                        html.Div([
                            html.Strong("Most Recent: "),
                            html.Span(f"{data['fair_value_gaps']['most_recent']['type'].title()} Gap" if data['fair_value_gaps']['most_recent'] else "None")
                        ]) if data['fair_value_gaps']['most_recent'] else None
                    ])
                ])
            ]
            timeframe_sections.extend(sections)
        
        return html.Div([
            html.H3(f"Technical Analysis for {symbol}"),
            *timeframe_sections
        ])
        
    except Exception as e:
        logger.error(f"Error in technical analysis: {str(e)}", exc_info=True)
        return html.Div(f"Error calculating technical analysis: {str(e)}")

class StockApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Data Viewer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize API client
        self.api = SchwabAPI()
        self.options_handler = SchwabOptionsHandler()
        
        # Initialize data storage
        self.minute_data = None
        self.daily_data = None
        self.options_data = {}
        
        # Create UI
        self.init_ui()
        
        # Set up timer for data updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(60000)  # Update every minute
        
        # Set up timer for options updates
        self.options_timer = QTimer()
        self.options_timer.timeout.connect(self.update_options_data)
        self.options_timer.start(1000)  # Update every second
        
        logger.info("Application initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        try:
            # Create central widget and layout
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Create input section
            input_layout = QHBoxLayout()
            self.symbol_input = QLineEdit()
            self.symbol_input.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
            self.fetch_button = QPushButton("Fetch Data")
            self.fetch_button.clicked.connect(self.fetch_data)
            input_layout.addWidget(QLabel("Symbol:"))
            input_layout.addWidget(self.symbol_input)
            input_layout.addWidget(self.fetch_button)
            layout.addLayout(input_layout)
            
            # Create tab widget
            self.tab_widget = QTabWidget()
            
            # Create minute data tab
            minute_tab = QWidget()
            minute_layout = QVBoxLayout(minute_tab)
            self.minute_table = QTableWidget()
            minute_layout.addWidget(self.minute_table)
            self.tab_widget.addTab(minute_tab, "Minute Data")
            
            # Create daily data tab
            daily_tab = QWidget()
            daily_layout = QVBoxLayout(daily_tab)
            self.daily_table = QTableWidget()
            daily_layout.addWidget(self.daily_table)
            self.tab_widget.addTab(daily_tab, "Daily Data")
            
            # Create options chain tab
            options_tab = QWidget()
            options_layout = QVBoxLayout(options_tab)
            
            # Create tables for calls and puts
            self.calls_table = QTableWidget()
            self.puts_table = QTableWidget()
            
            # Add tables to options layout
            options_layout.addWidget(QLabel("Calls"))
            options_layout.addWidget(self.calls_table)
            options_layout.addWidget(QLabel("Puts"))
            options_layout.addWidget(self.puts_table)
            
            self.tab_widget.addTab(options_tab, "Options Chain")
            
            layout.addWidget(self.tab_widget)
            
            # Create status bar
            self.statusBar().showMessage("Ready")
            
            logger.info("UI initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to initialize UI: {str(e)}")
    
    def fetch_data(self):
        """Fetch data for the entered symbol."""
        try:
            symbol = self.symbol_input.text().strip().upper()
            if not symbol:
                QMessageBox.warning(self, "Error", "Please enter a symbol")
                return
            
            logger.info(f"Fetching data for {symbol}")
            self.statusBar().showMessage(f"Fetching data for {symbol}...")
            
            # Initialize options handler if needed
            if not self.options_handler.is_stream_active():
                if not self.options_handler.initialize(self.api.client):
                    raise Exception("Failed to initialize options handler")
            
            # Start options stream
            if not self.options_handler.start_stream(symbol):
                raise Exception("Failed to start options stream")
            
            # Fetch minute data
            self.minute_data = self.api.get_minute_data(symbol)
            if self.minute_data is not None:
                self.update_minute_table()
            
            # Fetch daily data
            self.daily_data = self.api.get_daily_data(symbol)
            if self.daily_data is not None:
                self.update_daily_table()
            
            self.statusBar().showMessage(f"Data fetched for {symbol}")
            logger.info(f"Data fetch completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {str(e)}")
            self.statusBar().showMessage("Error fetching data")
    
    def update_data(self):
        """Update data periodically."""
        try:
            symbol = self.symbol_input.text().strip().upper()
            if not symbol:
                return
            
            logger.debug(f"Updating data for {symbol}")
            
            # Update minute data
            self.minute_data = self.api.get_minute_data(symbol)
            if self.minute_data is not None:
                self.update_minute_table()
            
            # Update daily data
            self.daily_data = self.api.get_daily_data(symbol)
            if self.daily_data is not None:
                self.update_daily_table()
            
            logger.debug(f"Data update completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating data: {e}", exc_info=True)
    
    def update_options_data(self):
        """Update options data from the stream handler."""
        try:
            if not self.options_handler.is_stream_active():
                logger.debug("Options stream is not active")
                return
            
            contracts = self.options_handler.get_contracts()
            if not contracts:
                logger.debug("No contracts received from stream handler")
                return
            
            logger.debug(f"Received {len(contracts)} contracts from stream handler")
            self.options_data = contracts
            self.update_options_tables()
            
        except Exception as e:
            logger.error(f"Error updating options data: {e}", exc_info=True)
    
    def update_minute_table(self):
        """Update the minute data table."""
        try:
            if self.minute_data is None or self.minute_data.empty:
                return
            
            self.minute_table.setRowCount(len(self.minute_data))
            self.minute_table.setColumnCount(len(self.minute_data.columns))
            self.minute_table.setHorizontalHeaderLabels(self.minute_data.columns)
            
            for i, (_, row) in enumerate(self.minute_data.iterrows()):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.minute_table.setItem(i, j, item)
            
            self.minute_table.resizeColumnsToContents()
            logger.debug("Minute data table updated")
            
        except Exception as e:
            logger.error(f"Error updating minute table: {e}", exc_info=True)
    
    def update_daily_table(self):
        """Update the daily data table."""
        try:
            if self.daily_data is None or self.daily_data.empty:
                return
            
            self.daily_table.setRowCount(len(self.daily_data))
            self.daily_table.setColumnCount(len(self.daily_data.columns))
            self.daily_table.setHorizontalHeaderLabels(self.daily_data.columns)
            
            for i, (_, row) in enumerate(self.daily_data.iterrows()):
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.daily_table.setItem(i, j, item)
            
            self.daily_table.resizeColumnsToContents()
            logger.debug("Daily data table updated")
            
        except Exception as e:
            logger.error(f"Error updating daily table: {e}", exc_info=True)
    
    def update_options_tables(self):
        """Update the options chain tables."""
        try:
            if not self.options_data:
                logger.debug("No options data available")
                return
            
            # Separate calls and puts
            calls = []
            puts = []
            
            for contract in self.options_data:
                if contract["contract_type"] == "C":
                    calls.append(contract)
                else:
                    puts.append(contract)
            
            logger.debug(f"Processing {len(calls)} calls and {len(puts)} puts")
            
            # Sort by strike price
            calls.sort(key=lambda x: x["strike"])
            puts.sort(key=lambda x: x["strike"])
            
            # Update calls table
            self.calls_table.setRowCount(len(calls))
            self.calls_table.setColumnCount(8)
            self.calls_table.setHorizontalHeaderLabels([
                "Strike", "Bid", "Ask", "Last", "Volume", "OI", "IV", "Delta"
            ])
            
            for i, contract in enumerate(calls):
                self.calls_table.setItem(i, 0, QTableWidgetItem(f"${contract['strike']:.2f}"))
                self.calls_table.setItem(i, 1, QTableWidgetItem(f"${contract['bid']:.2f}" if contract['bid'] != -999 else "N/A"))
                self.calls_table.setItem(i, 2, QTableWidgetItem(f"${contract['ask']:.2f}" if contract['ask'] != -999 else "N/A"))
                self.calls_table.setItem(i, 3, QTableWidgetItem(f"${contract['last']:.2f}" if contract['last'] != -999 else "N/A"))
                self.calls_table.setItem(i, 4, QTableWidgetItem(f"{contract['volume']:,}" if contract['volume'] != -999 else "N/A"))
                self.calls_table.setItem(i, 5, QTableWidgetItem(f"{contract['open_interest']:,}" if contract['open_interest'] != -999 else "N/A"))
                self.calls_table.setItem(i, 6, QTableWidgetItem(f"{contract['volatility']:.1%}" if contract['volatility'] != -999 else "N/A"))
                self.calls_table.setItem(i, 7, QTableWidgetItem(f"{contract['delta']:.3f}" if contract['delta'] != -999 else "N/A"))
            
            # Update puts table
            self.puts_table.setRowCount(len(puts))
            self.puts_table.setColumnCount(8)
            self.puts_table.setHorizontalHeaderLabels([
                "Strike", "Bid", "Ask", "Last", "Volume", "OI", "IV", "Delta"
            ])
            
            for i, contract in enumerate(puts):
                self.puts_table.setItem(i, 0, QTableWidgetItem(f"${contract['strike']:.2f}"))
                self.puts_table.setItem(i, 1, QTableWidgetItem(f"${contract['bid']:.2f}" if contract['bid'] != -999 else "N/A"))
                self.puts_table.setItem(i, 2, QTableWidgetItem(f"${contract['ask']:.2f}" if contract['ask'] != -999 else "N/A"))
                self.puts_table.setItem(i, 3, QTableWidgetItem(f"${contract['last']:.2f}" if contract['last'] != -999 else "N/A"))
                self.puts_table.setItem(i, 4, QTableWidgetItem(f"{contract['volume']:,}" if contract['volume'] != -999 else "N/A"))
                self.puts_table.setItem(i, 5, QTableWidgetItem(f"{contract['open_interest']:,}" if contract['open_interest'] != -999 else "N/A"))
                self.puts_table.setItem(i, 6, QTableWidgetItem(f"{contract['volatility']:.1%}" if contract['volatility'] != -999 else "N/A"))
                self.puts_table.setItem(i, 7, QTableWidgetItem(f"{contract['delta']:.3f}" if contract['delta'] != -999 else "N/A"))
            
            self.calls_table.resizeColumnsToContents()
            self.puts_table.resizeColumnsToContents()
            logger.debug("Options tables updated")
            
        except Exception as e:
            logger.error(f"Error updating options tables: {e}", exc_info=True)
    
    def closeEvent(self, event):
        """Handle application closure."""
        try:
            # Stop options stream
            if self.options_handler.is_stream_active():
                self.options_handler.stop_stream()
            
            # Stop timers
            self.update_timer.stop()
            self.options_timer.stop()
            
            logger.info("Application closed")
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during closure: {e}", exc_info=True)
            event.accept()

    def setup_options_chain_tab(self):
        """Set up the options chain tab with tables for calls and puts."""
        options_tab = QWidget()
        layout = QVBoxLayout()
        
        # Add controls at the top
        controls_layout = QHBoxLayout()
        
        # Symbol input
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter stock symbol")
        controls_layout.addWidget(self.symbol_input)
        
        # Fetch Data button
        fetch_button = QPushButton("Fetch Data")
        fetch_button.clicked.connect(self.fetch_data)
        controls_layout.addWidget(fetch_button)
        
        # Stream control buttons
        self.start_stream_button = QPushButton("Start Stream")
        self.start_stream_button.clicked.connect(self.start_options_stream)
        controls_layout.addWidget(self.start_stream_button)
        
        self.stop_stream_button = QPushButton("Stop Stream")
        self.stop_stream_button.clicked.connect(self.stop_options_stream)
        controls_layout.addWidget(self.stop_stream_button)
        
        # Add test button
        test_button = QPushButton("Test Stream")
        test_button.clicked.connect(self.test_options_stream)
        controls_layout.addWidget(test_button)
        
        layout.addLayout(controls_layout)
        
        # Rest of your existing setup code...
        
    def test_options_stream(self):
        """Test the options stream to examine field mappings."""
        try:
            symbol = self.symbol_input.text().strip().upper()
            if not symbol:
                symbol = "SPY"  # Default to SPY if no symbol entered
            
            logger.info(f"Testing options stream for {symbol}")
            if self.options_stream_handler:
                self.options_stream_handler.test_stream(symbol)
            else:
                logger.error("Options stream handler not initialized")
        except Exception as e:
            logger.error(f"Error testing options stream: {e}", exc_info=True)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
    if sys.argv[1:] == ['--gui']:
        app = QApplication(sys.argv)
        window = StockApp()
        window.show()
        sys.exit(app.exec()) 