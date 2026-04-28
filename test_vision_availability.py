import os
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ No API KEY found")
    exit(1)

client = Groq(api_key=api_key)

# 1. List all models explicitly
print("🔍 Listing available models...")
try:
    models = client.models.list()
    vision_models = [m.id for m in models.data if "vision" in m.id or "llava" in m.id]
    all_models = [m.id for m in models.data]
    
    print(f"Found {len(all_models)} models total.")
    print("Vision models found:", vision_models)
    
    if not vision_models:
        print("⚠️ NO VISION MODELS FOUND. This is likely the root cause.")
        print("All available models:", all_models)
except Exception as e:
    print(f"❌ Error listing models: {e}")

# 2. Try a test generation with 11b-vision (if it exists or just force it)
target_model = "llama-3.2-11b-vision-preview"
print(f"\n🧪 Testing {target_model}...")

# Tiny 1x1 transparent pixel for testing
dummy_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

try:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{dummy_image}"
                        }
                    }
                ]
            }
        ],
        model=target_model,
        max_tokens=10,
    )
    print("✅ Success! Response:", chat_completion.choices[0].message.content)
except Exception as e:
    print(f"❌ Failed to use {target_model}: {e}")
