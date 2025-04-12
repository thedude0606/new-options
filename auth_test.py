import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Get credentials from environment variables
APP_KEY = os.getenv('SCHWAB_APP_KEY')
APP_SECRET = os.getenv('SCHWAB_APP_SECRET')
CALLBACK_URL = os.getenv('SCHWAB_CALLBACK_URL')

def get_access_token():
    """
    Get an access token from Schwab API
    """
    auth_url = "https://api.schwab.com/v1/oauth2/token"
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    
    data = {
        'grant_type': 'client_credentials',
        'client_id': APP_KEY,
        'client_secret': APP_SECRET,
        'redirect_uri': CALLBACK_URL
    }
    
    try:
        response = requests.post(auth_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting access token: {e}")
        return None

if __name__ == "__main__":
    print("Attempting to authenticate with Schwab API...")
    token_response = get_access_token()
    
    if token_response:
        print("Authentication successful!")
        print("Token response:")
        print(json.dumps(token_response, indent=2))
    else:
        print("Authentication failed!") 