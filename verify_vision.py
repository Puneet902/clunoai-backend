import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key) if api_key else None

# 10x10 gray pixel (valid size and clearly not red/black)
pixel_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAAXNSR0IArs4c6QAAABJJREFUGFdj7OzsfMRAAowMDAwAcfEF9m9V8vQAAAAASUVORK5CYII="

model_id = "meta-llama/llama-4-scout-17b-16e-instruct"

results = []

if not client:
    results.append("❌ GROQ_API_KEY not found")
else:
    try:
        print(f"Testing {model_id}...")
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the content of this image in 1 word."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{pixel_b64}"
                            }
                        }
                    ]
                }
            ],
            model=model_id,
        )
        results.append(f"✅ {model_id}: {completion.choices[0].message.content}")
    except Exception as e:
        results.append(f"❌ {model_id}: {str(e)}")

with open("vision_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))
    
print("Done.")
