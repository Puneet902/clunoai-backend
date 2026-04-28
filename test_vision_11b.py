import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 1x1 transparent red pixel
dummy_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

def test_model(model_name):
    print(f"Testing {model_name}...")
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
            model=model_name,
            max_tokens=10,
        )
        print(f"Success with {model_name}: {chat_completion.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Failed {model_name}: {e}")
        return False

test_model("llama-3.2-11b-vision-preview")
