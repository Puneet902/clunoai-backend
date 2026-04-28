import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found in .env")
    exit(1)

client = Groq(api_key=api_key)

try:
    print("🔍 Fetching available Groq models...")
    models = client.models.list()
    
    output_path = "available_models.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for model in sorted(models.data, key=lambda x: x.id):
            f.write(f"{model.id}\n")
            
    print(f"✅ Successfully wrote {len(models.data)} models to {output_path}")
except Exception as e:
    print(f"❌ Error: {e}")
