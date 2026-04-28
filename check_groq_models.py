import os
from dotenv import load_dotenv
from groq import Groq

# Force load .env
load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")
print(f"DEBUG: API Key found: {'Yes' if api_key else 'No'}")

if not api_key:
    print("Error: No API key found in environment.")
    exit(1)

try:
    client = Groq(api_key=api_key)
    models = client.models.list()
    
    print("\nAvailable Models:")
    for m in models.data:
        print(f" - {m.id}")
        
except Exception as e:
    print(f"Error listing models: {e}")
