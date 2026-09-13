import os

from google.colab import userdata

# Get the key
api_key = userdata.get('GEMINI_API_KEY')

# Install localtunnel and run the app with the API key injected
#!npm install -g localtunnel
!GEMINI_API_KEY="{api_key}" python app.py & lt --port 8000
