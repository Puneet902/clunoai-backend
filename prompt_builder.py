"""
Prompt Builder Module
Builds structured prompts for the AI models with resume and JD context.
Implements mode-based question routing for human-like interview answers.
"""

import re

# Global storage for resume and JD
_resume_text = ""
_jd_text = ""

# Question Mode Constants
MODE_PERSONAL = "personal"
MODE_TECHNICAL = "technical"
MODE_BEHAVIORAL = "behavioral"
MODE_CODING = "coding"
MODE_SYSTEM_DESIGN = "system_design"
MODE_GREETING = "greeting"


def set_resume(text: str):
    """Store the resume text."""
    global _resume_text
    _resume_text = text


def set_jd(text: str):
    """Store the job description text."""
    global _jd_text
    _jd_text = text


def get_resume() -> str:
    """Get the stored resume text."""
    return _resume_text


def get_jd() -> str:
    """Get the stored JD text."""
    return _jd_text


def clear_context():
    """Clear resume and JD context."""
    global _resume_text, _jd_text
    _resume_text = ""
    _jd_text = ""


def classify_question(question: str) -> str:
    """
    Classify the question into one of five modes using rule-based logic.
    Returns: MODE_PERSONAL, MODE_TECHNICAL, MODE_BEHAVIORAL, MODE_CODING, or MODE_SYSTEM_DESIGN
    """
    q_lower = question.lower()
    
    # MODE_CODING: Coding / DSA triggers
    coding_triggers = [
        "write code", "write a function", "write a program", "implement", "solve", 
        "algorithm", "array", "linked list", "tree", "graph", "dp", "dynamic programming",
        "binary search", "sorting", "recursion", "data structure", "leetcode", "hackerrank",
        "dsa", "coding question", "code for", "solution for", "programming problem",
        "show code", "example code", "how to implement", "write some code", "coding", "programming"
    ]
    for t in coding_triggers:
        if t in q_lower: return MODE_CODING

    # MODE_SYSTEM_DESIGN: Scalability, architecture
    design_triggers = ["system design", "scalability", "architecture", "microservices", "load balancer", "database schema"]
    for t in design_triggers:
        if t in q_lower: return MODE_SYSTEM_DESIGN

    # MODE_BEHAVIORAL: Situational questions
    behavioral_triggers = ["tell me about a time", "situation where", "conflict", "difficult", "challenge you faced", "how did you handle"]
    for t in behavioral_triggers:
        if t in q_lower: return MODE_BEHAVIORAL

    # MODE_PERSONAL: Personal / Resume-based triggers
    personal_triggers = [
        "tell me about yourself", "your skills", "your experience", "your projects",
        "resume", "background", "why should we hire you", "introduce yourself",
        "about you", "what have you", "tell me about your project", "role in",
        "contribution", "worked on", "projects you've done", "achievements"
    ]
    for t in personal_triggers:
        if t in q_lower: return MODE_PERSONAL
    
    # MODE_GREETING: Small talk, greetings
    greeting_triggers = ["hello", "hi", "hey", "good morning", "good afternoon", "nice to meet you", "how are you"]
    if any(t in q_lower for t in greeting_triggers) and len(q_lower.split()) < 5:
        return MODE_GREETING

    # Default: MODE_TECHNICAL
    return MODE_TECHNICAL


def _build_resume_context() -> str:
    """Build a structured, concise resume context for injection."""
    if not _resume_text:
        return "No resume provided."
    return f"CANDIDATE RESUME DATA:\n{_resume_text[:2000]}"


# Core System Persona (UPGRADED for strictly First-Person Read-Ready Script Output)
CORE_SYSTEM_PERSONA = """You ARE the candidate being interviewed. You are NOT an AI assistant.

Rules for all your responses:
1. FIRST PERSON ONLY: Strictly use 'I', 'me', 'my', 'mine'. (e.g., "I worked on...", "My background is...", "I believe...").
2. YOUR IDENTITY: You are a professional engineer/candidate. You are describing YOUR OWN life and career based on the provided resume.
3. SCRIPT QUALITY: Generate a natural, "read-ready" script. Output ONLY the exact spoken words that the candidate will read out loud to the interviewer.
4. ABSOLUTELY NO METATEXT OR PREAMBLES: NEVER say "Here is a script", "Sure!", "Certainly", "Here is the answer", "As a candidate", "Question Mode", or "Hope this helps". Start your response IMMEDIATELY with the first spoken word.
5. NO THINKING TAGS: Never include <think>, </think>, or internal reasoning notes in your output.
6. NO MARKDOWN HEADINGS: Use bullet points (•) for key points, but NEVER use headings (# or ##).
7. BREVITY IS MANDATORY: Keep answers SHORT and punchy — 1 short intro sentence + 3 to 4 bullet points max. Every bullet must be concise (one line max)."""


def build_interview_prompt(question: str, language: str = "python") -> tuple[str, str]:
    """
    Build a prompt for high-quality, read-ready interview answers.
    """
    mode = classify_question(question)
    resume_context = _build_resume_context()
    
    system_prompt = CORE_SYSTEM_PERSONA
    
    if mode == MODE_CODING:
        system_prompt = """You are a senior engineer in a coding interview. Your goal is to provide a clear explanation followed by complete, working code implementations.

Your response MUST follow this EXACT STRUCTURE:

1. Approach (1 sentence + 2-3 bullet points max):
   • State the core idea and logic in one sentence.
   • Mention time/space complexity clearly (e.g., O(n) time, O(1) space).

2. Code Implementation (YOU MUST PROVIDE COMPLETE SOLUTIONS FOR ALL 4 LANGUAGES BELOW):
   • Ensure code is clean, idiomatic, and properly indented.
   • Use multi-line formatting for all blocks.

   ```python
   # Complete Python solution
   def solution():
       pass
   ```
   ```cpp
   // Complete C++ solution
   void solution() {
   }
   ```
   ```c
   // Complete C solution
   void solution() {
   }
   ```
   ```java
   // Complete Java solution
   class Solution {
       public void solution() {
       }
   }
   ```
   CRITICAL: You MUST provide the full implementation logic inside these blocks. Never leave them empty or as placeholders.

3. Key Points (2-3 bullet points max):
   • One-line explanation of the core logic.
   • Edge cases or gotchas if any."""

        user_prompt = f"Problem: {question}\n\nRespond as the candidate. Provide the Approach, then THE FULL CODE SOLUTIONS for all 4 languages (Python, C++, C, Java), then Key Points. Ensure the code is complete and correctly indented. Be concise but thorough in the code logic."

    elif mode == MODE_BEHAVIORAL or mode == MODE_PERSONAL:
        user_prompt = f"{resume_context}\n\nQuestion: \"{question}\"\n\nRespond as the candidate ('I'). One short intro sentence, then 3-4 bullet points using real details from the resume. Keep every bullet to one line. No long paragraphs."

    elif mode == MODE_SYSTEM_DESIGN:
        user_prompt = f"Question: \"{question}\"\n\nAnswer as the candidate/senior architect. One sentence overview, then 3-5 bullet points covering the key design decisions, scalability, and trade-offs. Keep it concise."

    elif mode == MODE_GREETING:
        user_prompt = f"Question: \"{question}\"\n\nRespond with a single short, professional, friendly greeting sentence as the candidate. Mention excitement to be here. Max 2 sentences."

    else: # TECHNICAL
        user_prompt = f"Question: \"{question}\"\n\nAnswer directly as the candidate. One sentence definition/summary, then 3-5 bullet points covering key concepts, use cases, or differences. No preambles, no long paragraphs."

    return system_prompt, user_prompt


def clean_filler_words(text: str) -> str:
    """
    Clean common filler words and repeated words from transcribed text.
    """
    if not text:
        return text
    
    # Remove common filler words
    fillers = [
        r'\b(uh|um|uhm|umm|er|err|ah|ahh|hmm|hm)\b',
        r'\b(like|you know|i mean|basically|actually|so yeah)\b',
    ]
    
    cleaned = text
    for pattern in fillers:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove repeated words (e.g., "the the" -> "the")
    cleaned = re.sub(r'\b(\w+)\s+\1\b', r'\1', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned
