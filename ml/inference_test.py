from unsloth import FastLanguageModel
import torch

# 1. Configuration
model_name = "lora_model" # Path to your trained adapters
max_seq_length = 2048

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# 3. Test Function
def generate_interview_answer(question, company="Google"):
    messages = [
        {"role": "system", "content": f"Classify as [personal/technical/coding]. For personal, ≤250 words. Tailor to {company}."},
        {"role": "user", "content": f"Question: {question}"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 512, use_cache = True)
    response = tokenizer.batch_decode(outputs)
    return response[0]

# 4. Example Run
if __name__ == "__main__":
    test_q = "Tell me about yourself and why you're a good fit for this role."
    print(f"❓ Question: {test_q}")
    answer = generate_interview_answer(test_q)
    print(f"✅ AI Response:\n{answer}")
