import requests

API_URL = "http://127.0.0.1:8000/eeg"

def send_eeg(features, state):
    try:
        response = requests.post(API_URL, json={
            "features": features,
            "state": state
        })
        return response.json()
    except:
        return {"error": "Failed to send"}