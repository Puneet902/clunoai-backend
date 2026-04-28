import requests
import base64
import json

# Dummy tiny wav content (just a few bytes, will trigger "Audio too short" but should pass Pydantic)
dummy_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x40\x1f\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00"
audio_b64 = base64.b64encode(dummy_audio).decode('utf-8')

url = "http://localhost:8000/transcribe-and-stream"
payload = {
    "audio_base_64": audio_b64,
    "model": "groq",
    "language": "python",
    "mode": "full"
}

print(f"Sending request to {url} with audio_base_64...")
try:
    response = requests.post(url, json=payload, stream=True)
    print(f"Status Code: {response.status_code}")
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
