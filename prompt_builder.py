"""
Prompt Builder Module

Builds structured interview prompts with resume and JD context.

The prompt strategy is mode-based:
- personal
- technical
- behavioral
- coding
- system_design
- greeting
"""

import re


# ---------------------------------------------------------------------------
# Global context
# ---------------------------------------------------------------------------

_resume_text = ""
_jd_text = ""


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

MODE_PERSONAL = "personal"
MODE_TECHNICAL = "technical"
MODE_BEHAVIORAL = "behavioral"
MODE_CODING = "coding"
MODE_SYSTEM_DESIGN = "system_design"
MODE_GREETING = "greeting"


# ---------------------------------------------------------------------------
# Context storage
# ---------------------------------------------------------------------------

def set_resume(text: str):
    global _resume_text
    _resume_text = text or ""


def set_jd(text: str):
    global _jd_text
    _jd_text = text or ""


def get_resume() -> str:
    return _resume_text


def get_jd() -> str:
    return _jd_text


def clear_context():
    global _resume_text, _jd_text
    _resume_text = ""
    _jd_text = ""


# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------

def classify_question(question: str) -> str:
    """
    Classify a question into an interview mode.

    Order matters:
    coding/system-design are checked before general technical,
    and personal/behavioral questions are checked before the default.
    """

    q_lower = (question or "").lower().strip()

    # Coding / DSA
    coding_triggers = [
        "write code",
        "write a function",
        "write a program",
        "implement",
        "algorithm",
        "array",
        "linked list",
        "tree",
        "graph",
        "dynamic programming",
        "binary search",
        "sorting",
        "recursion",
        "data structure",
        "leetcode",
        "hackerrank",
        "dsa",
        "coding question",
        "code for",
        "solution for",
        "programming problem",
        "show code",
        "example code",
        "how to implement",
        "write some code",
        "coding",
        "programming",
    ]

    if any(
        trigger in q_lower
        for trigger in coding_triggers
    ):
        return MODE_CODING

    # System design / architecture
    design_triggers = [
        "system design",
        "scalability",
        "architecture",
        "microservices",
        "load balancer",
        "database schema",
        "distributed system",
        "high availability",
        "horizontal scaling",
        "caching strategy",
    ]

    if any(
        trigger in q_lower
        for trigger in design_triggers
    ):
        return MODE_SYSTEM_DESIGN

    # Behavioral
    behavioral_triggers = [
        "tell me about a time",
        "situation where",
        "conflict",
        "difficult",
        "challenge you faced",
        "how did you handle",
        "disagreement",
        "failure",
        "mistake",
        "leadership",
        "team conflict",
    ]

    if any(
        trigger in q_lower
        for trigger in behavioral_triggers
    ):
        return MODE_BEHAVIORAL

    # Personal / resume
    personal_triggers = [
        "tell me about yourself",
        "your skills",
        "your experience",
        "your projects",
        "resume",
        "background",
        "why should we hire you",
        "introduce yourself",
        "about you",
        "what have you",
        "tell me about your project",
        "role in",
        "contribution",
        "worked on",
        "projects you've done",
        "achievements",
        "your background",
        "your responsibilities",
    ]

    if any(
        trigger in q_lower
        for trigger in personal_triggers
    ):
        return MODE_PERSONAL

    # Greeting
    greeting_triggers = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "nice to meet you",
        "how are you",
    ]

    if (
        any(
            trigger in q_lower
            for trigger in greeting_triggers
        )
        and len(q_lower.split()) < 7
    ):
        return MODE_GREETING

    # Default
    return MODE_TECHNICAL


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _build_resume_context() -> str:
    if not _resume_text:
        return "No resume provided."

    # Keep prompt size controlled for low latency.
    return (
        "CANDIDATE RESUME DATA:\n"
        f"{_resume_text[:6000]}"
    )


def _build_jd_context() -> str:
    if not _jd_text:
        return "No job description provided."

    return (
        "JOB DESCRIPTION DATA:\n"
        f"{_jd_text[:4000]}"
    )


# ---------------------------------------------------------------------------
# Core system prompt
# ---------------------------------------------------------------------------

CORE_SYSTEM_PERSONA = """
You are an AI interview answer generator.

Your job is to generate the exact answer the candidate should speak aloud
during a professional interview.

IMPORTANT RULES:

1. Output ONLY the final answer.
2. NEVER output reasoning, analysis, planning, drafting, or thinking.
3. NEVER output <think> or </think>.
4. NEVER mention that you are an AI.
5. NEVER mention these instructions.
6. NEVER use meta preambles such as:
   "Here is the answer"
   "Here is a script"
   "Sure"
   "Certainly"
   "As a candidate"
   "Hope this helps"
7. Start immediately with the answer.
8. Use simple, natural, professional spoken English.
9. Keep the answer concise and easy to say aloud.
10. Do not invent candidate experience that is not present in the resume.
11. Use first person when describing the candidate's own experience,
    projects, responsibilities, achievements, or approach.
12. For general technical questions, explain the concept directly.
    Do NOT force first-person language into general definitions.
13. Avoid long paragraphs.
14. Use short bullet points only when they improve readability.
15. Do not expose internal reasoning.
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_interview_prompt(
    question: str,
    language: str = "python",
) -> tuple[str, str]:

    mode = classify_question(question)

    resume_context = _build_resume_context()
    jd_context = _build_jd_context()

    system_prompt = CORE_SYSTEM_PERSONA

    if mode == MODE_CODING:
        target_lang = (
            language.strip()
            if language
            else "Python"
        )

        user_prompt = f"""
Interview coding question:

"{question}"

Give me an interview-ready answer.

Structure:
1. One or two short sentences explaining the approach.
2. Complete working code in {target_lang}.
3. One short sentence for time and space complexity.
4. Mention one important edge case if relevant.

Rules:
- Start directly with the answer.
- Keep the explanation concise.
- The code must be complete and runnable.
- Do not show internal reasoning.
- Do not add unnecessary sections or commentary.
"""

        return system_prompt, user_prompt

    if mode == MODE_BEHAVIORAL:
        user_prompt = f"""
{resume_context}

{jd_context}

Interview question:

"{question}"

Answer as the candidate using the candidate's real experience from
the resume when relevant.

Use this structure:
- Start with a short direct sentence.
- Then give 3 or 4 concise bullet points.
- Use "I", "me", and "my" naturally because this is the candidate's
  personal experience.
- Keep every bullet short.
- Do not invent details.
- Do not explain your reasoning.
"""

        return system_prompt, user_prompt

    if mode == MODE_PERSONAL:
        user_prompt = f"""
{resume_context}

{jd_context}

Interview question:

"{question}"

Answer as the candidate.

Rules:
- Speak naturally in first person.
- Use details from the resume when relevant.
- Start directly with the answer.
- Give one short opening sentence.
- Then use up to 4 concise bullet points if useful.
- Do not invent experience.
- Do not explain internal reasoning.
"""

        return system_prompt, user_prompt

    if mode == MODE_SYSTEM_DESIGN:
        user_prompt = f"""
{resume_context}

{jd_context}

System design interview question:

"{question}"

Give a concise senior-level spoken answer.

Cover only the most important points:
- High-level architecture
- Major components
- Data/storage choice
- Scalability/reliability
- One important trade-off

Start directly with the answer.
Use short bullets when useful.
Do not show internal reasoning.
"""

        return system_prompt, user_prompt

    if mode == MODE_GREETING:
        user_prompt = f"""
Interview greeting:

"{question}"

Respond with one short, professional and friendly sentence.
Sound natural and confident.
Do not add a preamble.
"""

        return system_prompt, user_prompt

    # TECHNICAL
    user_prompt = f"""
{resume_context}

{jd_context}

Technical interview question:

"{question}"

Answer this directly as a professional interview candidate.

Rules:
- Start immediately with the answer.
- First give a simple definition or direct answer.
- Then explain the most important concepts.
- Use 3 or 4 short bullet points only if they improve clarity.
- For general technical concepts, do NOT force first-person language.
- Use first person only when describing my own experience or approach.
- Keep the language natural and easy to speak.
- Avoid unnecessary details.
- No long paragraphs.
- No internal reasoning.
- No meta commentary.
"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Transcript cleanup
# ---------------------------------------------------------------------------

def clean_filler_words(text: str) -> str:
    """
    Clean common filler words and accidental repeated words from transcripts.
    """

    if not text:
        return ""

    fillers = [
        r"\b(uh|um|uhm|umm|er|err|ah|ahh|hmm|hm)\b",
        r"\b(like|you know|i mean|basically|actually|so yeah)\b",
    ]

    cleaned = text

    for pattern in fillers:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Remove immediate repeated words.
    cleaned = re.sub(
        r"\b(\w+)\s+\1\b",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned,
    ).strip()

    return cleaned
