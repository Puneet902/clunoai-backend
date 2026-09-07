"""
AI Router Module (STRICT GROQ-ONLY)
Handles routing strictly to Groq models as per user requirement.
"""

import os
import re
import json
from typing import Generator
from dotenv import load_dotenv
from groq import Groq
from prompt_builder import build_interview_prompt

def clean_script_output(text: str) -> str:
    """Clean reasoning tags (<think>...</think>), preambles, and meta chatter from output."""
    if not text:
        return text
    # Strip <think>...</think>
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip common meta preambles
    preambles = [
        r'^(here\s+is\s+(a\s+)?(natural\s+)?(spoken\s+)?(script|answer|response)[^\n]*\n?)',
        r'^(sure[!,.]?\s*here[^\n]*\n?)',
        r'^(certainly[!,.]?\s*here[^\n]*\n?)',
        r'^(as\s+a\s+candidate[^\n]*\n?)',
        r'^(hope\s+this\s+helps[^\n]*)'
    ]
    for p in preambles:
        cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned

# Load environment variables
load_dotenv()

# Model Constants for Optimization
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "qwen/qwen3.6-27b")
VERSATILE_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
VISION_MODEL = "llama-3.2-11b-vision-preview"  # For screen analysis


class AIRouter:
    """Routes questions strictly to Groq AI models."""
    
    def __init__(self):
        """Initialize the AI Router with Groq."""
        self._setup_groq()
    
    def _setup_groq(self):
        """Configure Groq API."""
        api_key = os.getenv("GROQ_API_KEY")
        if api_key and api_key != "your_groq_api_key_here":
            self.system_groq_client = Groq(api_key=api_key)
            self.system_groq_available = True
        else:
            self.system_groq_client = None
            self.system_groq_available = False
            print("[WARNING] AIRouter: GROQ_API_KEY is missing!")

    def generate_answer(self, question: str, model: str = "groq", api_key: str = None, language: str = "python", system_prompt: str = None) -> str:
        """
        Generate an answer using Groq (forced).
        """
        return self._generate_with_groq(question, api_key, model, language, system_prompt)

    def generate_answer_stream(self, question: str, model: str = "groq", api_key: str = None, language: str = "python", system_prompt: str = None) -> Generator[str, None, None]:
        """
        Generate a streaming answer using Groq (forced).
        """
        yield from self._generate_with_groq_stream(question, api_key, model, language, system_prompt)
    
    def generate_answer_from_image(self, image_data: str, prompt: str, api_key: str = None, model_id: str = "groq") -> str:
        """
        Generate answer from an image using Groq Vision.
        """
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        client = None
        if api_key:
            client = Groq(api_key=api_key)
        elif self.system_groq_available:
            client = self.system_groq_client
            
        if not client:
            return "Groq API key not configured."
            
        try:
            # Llama 3.2 90b Vision or Llama 3.2 11b Vision
            vision_model = "llama-3.2-90b-vision-preview"
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                model=vision_model,
                temperature=0.3,
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Groq Vision Error: {str(e)}"

    def generate_answer_from_image_stream(self, image_data: str, prompt: str, api_key: str = None, model_id: str = "groq") -> Generator[str, None, None]:
        """
        Stream answer from image using Groq.
        """
        res = self.generate_answer_from_image(image_data, prompt, api_key, model_id)
        yield res

    def _generate_with_groq(self, question: str, api_key: str = None, model_id: str = "llama-3.3-70b-versatile", language: str = "python", system_prompt: str = None) -> str:
        """Generate an answer using Groq API."""
        client = None
        if api_key:
            client = Groq(api_key=api_key)
        elif self.system_groq_available:
            client = self.system_groq_client
            
        if not client:
            return "Groq API key not configured."
        
        # Use healthy defaults for Groq
        groq_model = model_id if model_id and model_id != "groq" else DEFAULT_MODEL
            
        try:
            if system_prompt:
                sys_p, user_p = system_prompt, question
            else:
                sys_p, user_p = build_interview_prompt(question, language)
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p}
                ],
                model=groq_model,
                temperature=0.3,
                max_tokens=900,
            )
            raw_ans = chat_completion.choices[0].message.content
            return clean_script_output(raw_ans)
        except Exception as e:
            return f"Groq Error: {str(e)}"
    
    def _generate_with_groq_stream(self, question: str, api_key: str = None, model_id: str = "llama-3.3-70b-versatile", language: str = "python", system_prompt: str = None) -> Generator[str, None, None]:
        """Stream answer from Groq."""
        client = None
        if api_key:
            client = Groq(api_key=api_key)
        elif self.system_groq_available:
            client = self.system_groq_client
            
        if not client:
            yield "Groq API key not configured."
            return
        
        # Use healthy defaults for Groq
        groq_model = model_id if model_id and model_id != "groq" else DEFAULT_MODEL

        try:
            if system_prompt:
                sys_p, user_p = system_prompt, question
            else:
                sys_p, user_p = build_interview_prompt(question, language)
            
            stream = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p}
                ],
                model=groq_model,
                temperature=0.3,
                max_tokens=900,
                stream=True,
            )
            buffer = ""
            in_think = False
            first_chunk = True
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                buffer += content
                
                if "<think>" in buffer and not in_think:
                    in_think = True
                if "</think>" in buffer:
                    buffer = buffer.split("</think>")[-1]
                    in_think = False
                    
                if not in_think:
                    if first_chunk:
                        if len(buffer) > 40 or "\n" in buffer:
                            cleaned_part = clean_script_output(buffer)
                            first_chunk = False
                            buffer = ""
                            if cleaned_part:
                                yield cleaned_part
                    else:
                        yield buffer
                        buffer = ""
            if buffer and not in_think:
                if first_chunk:
                    yield clean_script_output(buffer)
                else:
                    yield buffer
        except Exception as e:
            yield f"Groq Stream Error: {str(e)}"

# Create singleton instance
ai_router = AIRouter()

def generate_answer(question: str, model: str = "groq", api_key: str = None, language: str = "python") -> str:
    return ai_router.generate_answer(question, model, api_key, language)

def generate_answer_stream(question: str, model: str = "groq", api_key: str = None, language: str = "python", system_prompt: str = None) -> Generator[str, None, None]:
    return ai_router.generate_answer_stream(question, model, api_key, language, system_prompt=system_prompt)

def generate_answer_from_image(image_data: str, prompt: str, model: str = "groq", api_key: str = None) -> str:
    return ai_router.generate_answer_from_image(image_data, prompt, api_key=api_key, model_id=model)

def generate_raw_prompt(prompt: str, model: str = "groq", api_key: str = None) -> str:
    """Send raw prompt strictly to Groq."""
    client = None
    if api_key:
        client = Groq(api_key=api_key)
    elif ai_router.system_groq_available:
        client = ai_router.system_groq_client
    
    if not client: return "Groq NOT configured."
    
    try:
        model_to_use = model if model and model != "groq" else DEFAULT_MODEL
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_to_use,
            temperature=0.3
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {str(e)}"
