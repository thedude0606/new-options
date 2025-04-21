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
import numpy as np
from advanced_technical_analysis import AdvancedTechnicalAnalysis

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('app')

# Initialize global variables
stored_minute_data = []
stored_daily_data = []
client = None
options_handler = None
latest_options_data = {}

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
    if data is None or (isinstance(data, pd.DataFrame) and data.empty):
        return []
    
    rows = []
    
    # If data is a DataFrame, convert it to a list of dictionaries
    if isinstance(data, pd.DataFrame):
        data = data.to_dict('records')
    
    for row in data:
        if isinstance(row, dict):
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
    dcc.Interval(id='interval-component', interval=60000),  # 60 second interval (1 minute)
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
        ]),
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
    """Update the data tables when the fetch button is clicked."""
    if not n_clicks or not symbol:
        raise PreventUpdate
        
    try:
        # Initialize client if needed
        global client, stored_minute_data, stored_daily_data
        if not client:
            client = initialize_client()
            
        # Fetch data
        logger.info(f"Fetching data for {symbol}")
        minute_data = get_minute_data(client, symbol)
        daily_data = get_daily_data(client, symbol)
        
        # Store the data globally
        stored_minute_data = minute_data
        stored_daily_data = daily_data
        
        # Process data for display
        minute_df = pd.DataFrame(minute_data)
        daily_df = pd.DataFrame(daily_data)
        
        # Format the data
        minute_table = process_data(minute_df)
        daily_table = process_data(daily_df)
        
        return minute_table, daily_table, f"Data fetched successfully for {symbol}"
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}", exc_info=True)
        return [], [], f"Error fetching data: {str(e)}"

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

def format_value(value):
    """Format numeric values for display."""
    if value is None:
        return "N/A"
    
    if isinstance(value, (np.float64, np.float32)):
        value = value.item()
    
    if isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.4f}"
        elif abs(value) < 1:
            return f"{value:.3f}"
        elif abs(value) < 100:
            return f"{value:.2f}"
        else:
            return f"{value:.1f}"
    return str(value)

# Add this helper function at the top level
def get_signal_style(signal):
    """Return CSS style based on signal type."""
    if signal == 'bullish':
        return {'color': '#4CAF50', 'fontWeight': 'bold'}  # Green
    elif signal == 'bearish':
        return {'color': '#f44336', 'fontWeight': 'bold'}  # Red
    elif signal == 'neutral':
        return {'color': '#9E9E9E', 'fontWeight': 'normal'}  # Gray
    else:
        return {}

def get_timeframe_params(timeframe):
    """Get the parameters for each timeframe based on specifications."""
    params = {
        '1min': {
            'trend_bars': 30,  # 30 bars for trend following
            'momentum_bars': 14,  # RSI 14 uses last 14 min
            'volatility_bars': 20,  # BB 20 & ATR 14
            'volume_bars': 14,  # MFI 14 (14 min)
            'pattern_bars': 10,  # Last 10 bars
            'fvg_bars': 5  # Last 5 bars for gaps
        },
        '15min': {
            'trend_bars': 16,  # 16 bars (<4 hr)
            'momentum_bars': 14,  # RSI 14 (last 3½ hr)
            'volatility_bars': 20,  # BB 20 & ATR 14
            'volume_bars': 14,  # MFI 14 (3½ hr)
            'pattern_bars': 10,  # Last 10 bars
            'fvg_bars': 5  # Last 5 bars
        },
        '1h': {
            'trend_bars': 24,  # 24 bars (~1 day)
            'momentum_bars': 14,  # RSI 14 (last 14 hr)
            'volatility_bars': 20,  # BB 20 & ATR 14
            'volume_bars': 14,  # MFI 14 (14 hr)
            'pattern_bars': 20,  # Last 20 bars
            'fvg_bars': 10  # Last 10 bars
        },
        'daily': {
            'trend_bars': 20,  # 20 bars (~1 month)
            'momentum_bars': 14,  # RSI 14 (14 days)
            'volatility_bars': 20,  # BB 20 & ATR 14
            'volume_bars': 14,  # MFI 14 (14 days)
            'pattern_bars': 20,  # Last 20 bars
            'fvg_bars': 10  # Last 10 bars
        }
    }
    return params.get(timeframe, params['daily'])

@callback(
    Output('technical-analysis-output', 'children'),
    [Input('symbol-input', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_technical_analysis(symbol, n_intervals):
    """Update the technical analysis when the symbol changes or on interval."""
    if not symbol:
        raise PreventUpdate
        
    try:
        global stored_minute_data, stored_daily_data
        
        if not stored_minute_data or not stored_daily_data:
            logger.warning("No data available for technical analysis")
            return html.Div("Please fetch data first using the 'Fetch Data' button.")
            
        logger.info(f"Calculating technical analysis for {symbol}")
        
        # Convert data to DataFrame
        minute_df = pd.DataFrame(stored_minute_data)
        minute_df['datetime'] = pd.to_datetime(minute_df['datetime'], unit='ms')
        minute_df.set_index('datetime', inplace=True)
        minute_df = minute_df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }).sort_index()
        
        timeframes = {
            '1min': minute_df,
            '15min': minute_df.resample('15min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna(),
            '1h': minute_df.resample('1h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'
            }).dropna(),
            'daily': pd.DataFrame(stored_daily_data).assign(
                datetime=lambda x: pd.to_datetime(x['datetime'], unit='ms')
            ).set_index('datetime').rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            }).sort_index()
        }
        
        all_sections = []
        
        for timeframe, df in timeframes.items():
            try:
                params = get_timeframe_params(timeframe)
                ta = AdvancedTechnicalAnalysis(df, timeframe=timeframe)
                timeframe_sections = []
                
                # Get current values
                close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2] if len(df) > 1 else close
                
                # Trend Indicators (using specified bars for each timeframe)
                trend_lookback = params['trend_bars']
                sma = ta.sma(trend_lookback)
                ema = ta.ema(trend_lookback)
                macd = ta.macd()
                macd_signal = ta.macd_signal()
                
                trend_signals = {
                    'SMA': 'bullish' if close > sma else 'bearish',
                    'EMA': 'bullish' if close > ema else 'bearish',
                    'MACD': 'bullish' if macd > macd_signal else 'bearish'
                }
                
                trend_section = html.Div([
                    html.H4(f"Trend Indicators (last {trend_lookback} bars)"),
                    html.Div([
                        html.P([
                            f"SMA ({trend_lookback}): ", 
                            html.Span(f"{format_value(sma)} ", style=get_signal_style(trend_signals['SMA'])),
                            html.Span(f"({trend_signals['SMA']})", style=get_signal_style(trend_signals['SMA']))
                        ]),
                        html.P([
                            f"EMA ({trend_lookback}): ",
                            html.Span(f"{format_value(ema)} ", style=get_signal_style(trend_signals['EMA'])),
                            html.Span(f"({trend_signals['EMA']})", style=get_signal_style(trend_signals['EMA']))
                        ]),
                        html.P([
                            "MACD: ",
                            html.Span(f"{format_value(macd)} ", style=get_signal_style(trend_signals['MACD'])),
                            html.Span(f"({trend_signals['MACD']})", style=get_signal_style(trend_signals['MACD']))
                        ])
                    ])
                ])
                timeframe_sections.append(trend_section)
                
                # Momentum Indicators (using specified lookback)
                mom_lookback = params['momentum_bars']
                rsi = ta.rsi(mom_lookback)
                stoch_k = ta.stoch_k()
                stoch_d = ta.stoch_d()
                
                momentum_signals = {
                    'RSI': 'bullish' if rsi < 30 else 'bearish' if rsi > 70 else 'neutral',
                    'Stochastic': 'bullish' if stoch_k > stoch_d and stoch_k < 20 else 'bearish' if stoch_k < stoch_d and stoch_k > 80 else 'neutral'
                }
                
                momentum_section = html.Div([
                    html.H4(f"Momentum Indicators (RSI {mom_lookback})"),
                    html.Div([
                        html.P([
                            f"RSI ({mom_lookback}): ",
                            html.Span(f"{format_value(rsi)} ", style=get_signal_style(momentum_signals['RSI'])),
                            html.Span(f"({momentum_signals['RSI']})", style=get_signal_style(momentum_signals['RSI']))
                        ]),
                        html.P([
                            "Stochastic %K/%D: ",
                            html.Span(f"{format_value(stoch_k)}/{format_value(stoch_d)} ", style=get_signal_style(momentum_signals['Stochastic'])),
                            html.Span(f"({momentum_signals['Stochastic']})", style=get_signal_style(momentum_signals['Stochastic']))
                        ])
                    ])
                ])
                timeframe_sections.append(momentum_section)
                
                # Volatility Indicators (using specified lookback)
                vol_lookback = params['volatility_bars']
                bb_upper = ta.bollinger_upper()
                bb_middle = ta.bollinger_middle()
                bb_lower = ta.bollinger_lower()
                atr = ta.atr()
                
                # Calculate ATR trend
                tr_current = max(df['High'].iloc[-1] - df['Low'].iloc[-1],
                               abs(df['High'].iloc[-1] - df['Close'].iloc[-2]),
                               abs(df['Low'].iloc[-1] - df['Close'].iloc[-2]))
                tr_prev = max(df['High'].iloc[-2] - df['Low'].iloc[-2],
                            abs(df['High'].iloc[-2] - df['Close'].iloc[-3]),
                            abs(df['Low'].iloc[-2] - df['Close'].iloc[-3]))
                
                volatility_signals = {
                    'Bollinger': 'bullish' if close < bb_lower else 'bearish' if close > bb_upper else 'neutral',
                    'ATR': 'high' if tr_current > tr_prev else 'low'
                }
                
                volatility_section = html.Div([
                    html.H4(f"Volatility Indicators (BB {vol_lookback})"),
                    html.Div([
                        html.P([
                            "Bollinger Bands: ",
                            html.Span(f"Upper: {format_value(bb_upper)}, Middle: {format_value(bb_middle)}, Lower: {format_value(bb_lower)} ", 
                                    style=get_signal_style(volatility_signals['Bollinger'])),
                            html.Span(f"({volatility_signals['Bollinger']})", style=get_signal_style(volatility_signals['Bollinger']))
                        ]),
                        html.P([
                            "ATR: ",
                            html.Span(f"{format_value(atr)} ", style={'color': '#2196F3' if volatility_signals['ATR'] == 'high' else '#9E9E9E'}),
                            html.Span(f"({volatility_signals['ATR']} volatility)", style={'color': '#2196F3' if volatility_signals['ATR'] == 'high' else '#9E9E9E'})
                        ])
                    ])
                ])
                timeframe_sections.append(volatility_section)
                
                # Volume Indicators (using specified lookback)
                vol_bars = params['volume_bars']
                obv = ta.obv()
                vol_sma = ta.volume_sma(vol_bars)
                
                # Calculate volume trend
                curr_vol = df['Volume'].iloc[-1]
                prev_vol = df['Volume'].iloc[-2]
                vol_trend = curr_vol > prev_vol
                price_trend = close > prev_close
                
                volume_signals = {
                    'OBV': 'bullish' if vol_trend and price_trend else 'bearish'
                }
                
                volume_section = html.Div([
                    html.H4(f"Volume Indicators (last {vol_bars} bars)"),
                    html.Div([
                        html.P([
                            "OBV: ",
                            html.Span(f"{format_value(obv)} ", style=get_signal_style(volume_signals['OBV'])),
                            html.Span(f"({volume_signals['OBV']})", style=get_signal_style(volume_signals['OBV']))
                        ]),
                        html.P(f"Volume SMA: {format_value(vol_sma)}")
                    ])
                ])
                timeframe_sections.append(volume_section)
                
                # Pattern Recognition (using specified lookback)
                pattern_bars = params['pattern_bars']
                patterns = {
                    'Doji': ('indecision', ta.doji()),
                    'Hammer': ('bullish', ta.hammer()),
                    'Engulfing': ('bullish' if ta.engulfing() else 'bearish', ta.engulfing()),
                    'Morning Star': ('bullish', ta.morning_star()),
                    'Evening Star': ('bearish', ta.evening_star()),
                    'Three White Soldiers': ('bullish', ta.three_white_soldiers()),
                    'Three Black Crows': ('bearish', ta.three_black_crows())
                }
                
                pattern_section = html.Div([
                    html.H4(f"Pattern Recognition (last {pattern_bars} bars)"),
                    html.Div([
                        html.P([
                            f"{pattern}: ",
                            html.Span(
                                f"{'Yes' if present else 'No'} ",
                                style=get_signal_style(signal) if present else {}
                            ),
                            html.Span(
                                f"({signal})" if present else "",
                                style=get_signal_style(signal) if present else {}
                            )
                        ]) for pattern, (signal, present) in patterns.items()
                    ])
                ])
                timeframe_sections.append(pattern_section)
                
                # Fair Value Gap Analysis
                try:
                    fvg_bars = params['fvg_bars']
                    recent_data = df.iloc[-fvg_bars:]
                    
                    fvg_gaps = []
                    for i in range(1, len(recent_data)):
                        curr_low = recent_data['Low'].iloc[i]
                        curr_high = recent_data['High'].iloc[i]
                        prev_low = recent_data['Low'].iloc[i-1]
                        prev_high = recent_data['High'].iloc[i-1]
                        
                        # Check for bullish gap (support)
                        if curr_low > prev_high:
                            fvg_gaps.append({
                                'type': 'bullish',
                                'upper': curr_low,
                                'lower': prev_high,
                                'time': recent_data.index[i]
                            })
                        
                        # Check for bearish gap (resistance)
                        elif curr_high < prev_low:
                            fvg_gaps.append({
                                'type': 'bearish',
                                'upper': prev_low,
                                'lower': curr_high,
                                'time': recent_data.index[i]
                            })
                    
                    fvg_section = html.Div([
                        html.H4(f"Fair Value Gaps (last {fvg_bars} bars)"),
                        html.Div([
                            html.P("Recent Fair Value Gaps:"),
                            html.Div([
                                html.P([
                                    f"Gap at {gap['time'].strftime('%Y-%m-%d %H:%M')}: ",
                                    html.Span(
                                        f"{gap['type'].capitalize()} Gap - Zone: {format_value(gap['lower'])} to {format_value(gap['upper'])} ",
                                        style=get_signal_style(gap['type'])
                                    ),
                                    html.Span(
                                        f"({'Support' if gap['type'] == 'bullish' else 'Resistance'})",
                                        style=get_signal_style(gap['type'])
                                    )
                                ]) for gap in fvg_gaps
                            ]) if fvg_gaps else html.P("No Fair Value Gaps detected in recent price action")
                        ])
                    ])
                    timeframe_sections.append(fvg_section)
                    
                except Exception as e:
                    logger.error(f"Error calculating Fair Value Gaps: {str(e)}", exc_info=True)
                    timeframe_sections.append(html.Div([
                        html.H4("Fair Value Gaps (FVG)"),
                        html.P(f"Error calculating Fair Value Gaps: {str(e)}")
                    ]))
                
                # Add timeframe section
                all_sections.append(html.Div([
                    html.H3(f"{timeframe} Timeframe", style={
                        'marginTop': '30px',
                        'marginBottom': '20px',
                        'borderBottom': '2px solid #ccc'
                    }),
                    html.Div(timeframe_sections, style={
                        'marginLeft': '20px'
                    })
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'marginBottom': '20px'
                }))
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {timeframe} timeframe: {str(e)}", exc_info=True)
                all_sections.append(html.Div([
                    html.H3(f"{timeframe} Timeframe"),
                    html.P(f"Error calculating indicators: {str(e)}")
                ]))
        
        logger.info("Technical analysis calculation completed")
        return html.Div(all_sections)
        
    except Exception as e:
        logger.error(f"Error calculating technical analysis: {str(e)}", exc_info=True)
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