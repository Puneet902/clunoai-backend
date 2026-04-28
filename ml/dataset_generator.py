import json
import os
import random

# Categories for classification
CATEGORIES = ["personal", "technical", "coding"]

# Sample questions for synthetic data generation
SAMPLE_QUESTIONS = {
    "personal": [
        "Tell me about yourself.",
        "What are your strengths and weaknesses?",
        "Why do you want to work for {company}?",
        "Tell me about a project you worked on.",
        "How do you handle conflict in a team?",
        "Where do you see yourself in five years?",
        "What is your greatest achievement?"
    ],
    "technical": [
        "Explain the difference between a process and a thread.",
        "How does a hash map work?",
        "What is virtual memory?",
        "Explain the TCP/IP stack.",
        "What are the ACID properties in databases?",
        "Explain the difference between SQL and NoSQL.",
        "How does a load balancer work?"
    ],
    "coding": [
        "Write a function to reverse a linked list.",
        "Implement a binary search algorithm.",
        "Find the longest common subsequence of two strings.",
        "Solve the two-sum problem using a hash map.",
        "Write a program to check if a string is a palindrome.",
        "Implement a merge sort algorithm.",
        "Write code to find the shortest path in a graph."
    ]
}

def generate_dataset(output_file="dataset.jsonl", num_examples=1000, company="Tech Corp"):
    """Generate a synthetic dataset in chat format for QLoRA fine-tuning."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in range(num_examples):
            category = random.choice(CATEGORIES)
            question_template = random.choice(SAMPLE_QUESTIONS[category])
            question = question_template.replace("{company}", company)
            
            # Placeholder for AI-generated answers
            # In a real scenario, you'd use a powerful LLM to generate these.
            answer_placeholder = f"This is a tailored {category} answer for {company}. "
            if category == "personal":
                answer_placeholder += "It is concise and stays under 250 words, highlighting innovation and culture fit."
            elif category == "coding":
                answer_placeholder += "It includes a clean approach, multi-line code, and line-by-line explanation."
            else:
                answer_placeholder += "It jump directly into the technical explanation without preamble."

            example = {
                "instruction": f"Classify as [personal/technical/coding]. For personal, ≤250 words. Tailor to {company}.",
                "input": f"Question: {question}",
                "output": f"Type: {category}. Answer: {answer_placeholder}"
            }
            
            # Format for SFTTrainer (ChatML or similar)
            chat_format = {
                "messages": [
                    {"role": "system", "content": f"Classify as [personal/technical/coding]. For personal, ≤250 words. Tailor to {company}."},
                    {"role": "user", "content": f"Question: {question}"},
                    {"role": "assistant", "content": f"Type: {category}. Answer: {answer_placeholder}"}
                ]
            }
            
            f.write(json.dumps(chat_format) + "\n")

    print(f"✅ Generated {num_examples} examples in {output_file}")

if __name__ == "__main__":
    generate_dataset()
