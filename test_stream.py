from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
import logging
import os
from dotenv import load_dotenv
import schwabdev
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_stream')

# Load environment variables
load_dotenv()
app_key = os.getenv('SCHWAB_APP_KEY')
app_secret = os.getenv('SCHWAB_APP_SECRET')

# Initialize client
client = schwabdev.Client(app_key=app_key, app_secret=app_secret)

# Initialize app
app = Dash(__name__)

# Simple layout
app.layout = html.Div([
    html.H1("Stream Test"),
    
    # Controls
    html.Div([
        html.Button('Start Stream', id='start-btn', n_clicks=0),
        html.Button('Stop Stream', id='stop-btn', n_clicks=0),
        html.Div(id='status')
    ]),
    
    # Data display
    html.Pre(id='raw-data'),
    
    # Update interval
    dcc.Interval(id='update-interval', interval=1000)
])

# Global variables for stream management
stream_active = False
latest_data = None
stream_lock = threading.Lock()

def on_message(message):
    """Handle incoming stream messages."""
    global latest_data
    with stream_lock:
        latest_data = message
        logger.debug(f"Received message: {message}")

@callback(
    [Output('status', 'children'),
     Output('raw-data', 'children')],
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('update-interval', 'n_intervals')]
)
def manage_stream(start_clicks, stop_clicks, n_intervals):
    """Manage the stream and display data."""
    global stream_active, latest_data
    
    # Handle button clicks
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'start-btn' and start_clicks:
            try:
                # Start stream for SPY options
                success = client.level_one_option_stream(
                    symbols=['SPY'],
                    fields=["1", "2", "3", "4", "8", "9", "10", "21", "28", "29", "30", "31"],
                    callback=on_message
                )
                
                if success:
                    stream_active = True
                    return "Stream started", "Waiting for data..."
                else:
                    return "Failed to start stream", ""
                    
            except Exception as e:
                logger.error(f"Error starting stream: {e}")
                return f"Error: {str(e)}", ""
                
        elif trigger_id == 'stop-btn' and stop_clicks:
            try:
                client.stop_streams()
                stream_active = False
                return "Stream stopped", ""
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
                return f"Error: {str(e)}", ""
    
    # Handle interval updates
    if stream_active:
        with stream_lock:
            if latest_data:
                return "Stream active", str(latest_data)
            return "Stream active - waiting for data", "No data yet"
    
    return "Stream inactive", ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)  # Use a different port than main app 