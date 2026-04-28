"""
Direct test: Does Gemini actually generate code with our prompt?
Run: python test_gemini_code.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Test 1: Check what build_interview_prompt produces for a coding question
from prompt_builder import build_interview_prompt, classify_question

test_q = "write code to reverse a linked list in python"
mode = classify_question(test_q)
print(f"=== Test 1: Classification ===")
print(f"Question: {test_q}")
print(f"Mode: {mode}")
print()

system_prompt, user_prompt = build_interview_prompt(test_q)
print(f"=== Test 2: Generated Prompts ===")
print(f"--- SYSTEM PROMPT ---")
print(system_prompt)
print(f"--- USER PROMPT ---")
print(user_prompt)
print()

# Test 2: Try calling Gemini directly
print(f"=== Test 3: Direct Gemini API Call ===")
try:
    import google.generativeai as genai
    
    # Try to get API key from .env or environment
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("No GEMINI_API_KEY in .env - checking if user provides via frontend...")
        print("Enter your Gemini API key to test (or press Enter to skip): ", end="")
        api_key = input().strip()
        
    if api_key:
        genai.configure(api_key=api_key)
        
        # Test with gemini-2.0-flash (what user likely selects)
        model_name = "gemini-2.0-flash"
        print(f"\nTesting with model: {model_name}")
        
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
            
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                )
            )
            
            print(f"\n--- GEMINI RESPONSE ---")
            print(response.text)
            print(f"--- END RESPONSE ---")
            print(f"\nResponse length: {len(response.text)} chars")
            print(f"Contains code block: {'```' in response.text}")
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            
            # Try fallback model
            model_name = "gemini-1.5-flash"
            print(f"\nRetrying with fallback model: {model_name}")
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_prompt
                )
                response = model.generate_content(
                    user_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=2048,
                    )
                )
                print(f"\n--- GEMINI RESPONSE ---")
                print(response.text)
                print(f"--- END RESPONSE ---")
                print(f"\nResponse length: {len(response.text)} chars")
                print(f"Contains code block: {'```' in response.text}")
            except Exception as e2:
                print(f"ERROR with {model_name}: {e2}")
    else:
        print("No API key provided, skipping direct test.")
        
except ImportError as e:
    print(f"Import error: {e}")

# Test 3: Check what the streaming path looks like
print("\n=== Test 4: Verify stream route ===")
from ai_router import AIRouter
router = AIRouter()
print(f"Gemini available (system): {router.system_gemini_available}")
print(f"Groq available (system): {router.system_groq_available}")
