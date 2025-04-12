from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import pandas as pd
from datetime import datetime
import os
from get_historical_data import get_minute_data, get_daily_data
from dotenv import load_dotenv
import schwabdev
import logging
from dash.exceptions import PreventUpdate

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dash_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize Dash app
logger.info("Initializing Dash application")
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Schwab Stock Data Viewer"),
    
    html.Div([
        html.Label("Enter Stock Symbol:"),
        dcc.Input(id='symbol-input', type='text', value='', placeholder='e.g., AAPL'),
        html.Button('Fetch Data', id='fetch-button', n_clicks=0),
    ], style={'margin': '20px'}),
    
    html.Div(id='status-message', style={'margin': '20px', 'color': 'blue'}),
    
    # Tabs
    dcc.Tabs([
        dcc.Tab(label='Minute Data (90 Days)', children=[
            dash_table.DataTable(
                id='minute-data-table',
                columns=[
                    {'name': 'Date/Time', 'id': 'datetime'},
                    {'name': 'Open', 'id': 'open'},
                    {'name': 'High', 'id': 'high'},
                    {'name': 'Low', 'id': 'low'},
                    {'name': 'Close', 'id': 'close'},
                    {'name': 'Volume', 'id': 'volume'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'center',
                    'minWidth': '100px',
                    'maxWidth': '180px',
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                sort_action='native',
                page_size=20,
                filter_action='native'
            )
        ]),
        dcc.Tab(label='Daily Data (365 Days)', children=[
            dash_table.DataTable(
                id='daily-data-table',
                columns=[
                    {'name': 'Date/Time', 'id': 'datetime'},
                    {'name': 'Open', 'id': 'open'},
                    {'name': 'High', 'id': 'high'},
                    {'name': 'Low', 'id': 'low'},
                    {'name': 'Close', 'id': 'close'},
                    {'name': 'Volume', 'id': 'volume'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'center',
                    'minWidth': '100px',
                    'maxWidth': '180px',
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                sort_action='native',
                page_size=20,
                filter_action='native'
            )
        ]),
    ])
])

def initialize_client():
    logger.info("Initializing Schwab client")
    load_dotenv()
    logger.debug("Environment variables loaded")
    
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
    
    try:
        client = schwabdev.Client(app_key, app_secret, callback_url)
        logger.info("Schwab client initialized successfully")
        return client
    except Exception as e:
        logger.error("Failed to initialize Schwab client", exc_info=True)
        raise

def process_data(data):
    if not data:
        logger.warning("No data to process")
        return []
    
    try:
        logger.debug(f"Processing {len(data)} records")
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Round numeric columns to 2 decimal places
        numeric_columns = ['open', 'high', 'low', 'close']
        df[numeric_columns] = df[numeric_columns].round(2)
        
        logger.debug(f"Data processing completed. First timestamp: {df['datetime'].iloc[0] if not df.empty else 'No data'}")
        return df.to_dict('records')
        
    except Exception as e:
        logger.error("Error processing data", exc_info=True)
        return []

@callback(
    [Output('minute-data-table', 'data'),
     Output('daily-data-table', 'data'),
     Output('status-message', 'children')],
    [Input('fetch-button', 'n_clicks')],
    [State('symbol-input', 'value')]
)
def update_tables(n_clicks, symbol):
    if not symbol or n_clicks == 0:
        logger.debug("No symbol provided or button not clicked")
        raise PreventUpdate
    
    symbol = symbol.upper()
    logger.info(f"Processing request for symbol: {symbol}")
    status_messages = []
    
    try:
        # Initialize client
        client = initialize_client()
        
        # Fetch minute data
        logger.info(f"Fetching minute data for {symbol}")
        minute_data = get_minute_data(client, symbol)
        status_messages.append(f"Fetched minute data for {symbol}")
        
        # Fetch daily data
        logger.info(f"Fetching daily data for {symbol}")
        daily_data = get_daily_data(client, symbol)
        status_messages.append(f"Fetched daily data for {symbol}")
        
        # Process and return the data
        minute_processed = process_data(minute_data)
        daily_processed = process_data(daily_data)
        
        logger.info(f"Data processing completed for {symbol}")
        return (
            minute_processed,
            daily_processed,
            "; ".join(status_messages)
        )
        
    except Exception as e:
        error_msg = f"Error fetching data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [], [], error_msg

if __name__ == '__main__':
    logger.info("Starting Dash application server")
    app.run(debug=True, port=8050) 