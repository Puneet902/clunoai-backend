"""
AI Router Module - STRICT GROQ-ONLY

Handles routing to Groq models and guarantees that reasoning content
is not exposed to the interview user.

Primary model:
    qwen/qwen3.6-27b

For interview answers:
    reasoning_format="hidden"
    reasoning_effort="none"
"""

import os
import re
from typing import Generator

from dotenv import load_dotenv
from groq import Groq

from prompt_builder import build_interview_prompt


load_dotenv()


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b",
)

FAST_MODEL = os.getenv(
    "GROQ_FAST_MODEL",
    "qwen/qwen3.6-27b",
)

VERSATILE_MODEL = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.6-27b",
)

VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "llama-3.2-90b-vision-preview",
)


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------

def clean_script_output(text: str) -> str:
    """
    Final safety cleanup.

    Groq is configured with reasoning_format="hidden", so normal responses
    should already contain only the final answer. This cleaner is retained
    as a defensive fallback.
    """

    if not text:
        return ""

    cleaned = text

    # Remove Qwen raw reasoning if it ever appears.
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove accidental reasoning/meta sections at the beginning.
    cleaned = re.sub(
        r"^\s*(thinking\s+process|analysis|reasoning)\s*:?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Remove common meta preambles.
    preambles = [
        r"^\s*here\s+is\s+(a\s+)?(natural\s+)?(spoken\s+)?(script|answer|response)\s*:?\s*",
        r"^\s*sure\s*[!,.]?\s*",
        r"^\s*certainly\s*[!,.]?\s*",
        r"^\s*as\s+a\s+candidate\s*[,:]?\s*",
        r"^\s*hope\s+this\s+helps\s*[!,.]?\s*",
    ]

    for pattern in preambles:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

    # Remove accidental markdown headings.
    cleaned = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        cleaned,
        flags=re.MULTILINE,
    )

    # Avoid excessive blank lines.
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


class StreamSanitizer:
    """
    Defensive streaming filter.

    Normally reasoning_format="hidden" means this never needs to do anything.
    If a provider/model accidentally sends <think> blocks, they are suppressed
    without leaking them to the frontend.
    """

    def __init__(self):
        self.buffer = ""
        self.in_think = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""

        self.buffer += chunk
        output_parts = []

        while self.buffer:
            if self.in_think:
                end = re.search(
                    r"</think>",
                    self.buffer,
                    flags=re.IGNORECASE,
                )

                if not end:
                    # Keep a small tail in case </think> is split across chunks.
                    self.buffer = self.buffer[-20:]
                    break

                self.buffer = self.buffer[
                    end.end():
                ]
                self.in_think = False
                continue

            start = re.search(
                r"<think>",
                self.buffer,
                flags=re.IGNORECASE,
            )

            if not start:
                # Keep a small suffix so a split "<think>" is handled.
                if len(self.buffer) > 20:
                    output_parts.append(
                        self.buffer[:-20]
                    )
                    self.buffer = self.buffer[-20:]

                break

            # Emit everything before <think>.
            if start.start() > 0:
                output_parts.append(
                    self.buffer[:start.start()]
                )

            self.buffer = self.buffer[
                start.end():
            ]
            self.in_think = True

        return "".join(output_parts)

    def flush(self) -> str:
        """
        Flush remaining safe text.

        If a malformed/unclosed <think> is present, it is intentionally
        discarded rather than exposed.
        """

        if self.in_think:
            return ""

        remaining = self.buffer
        self.buffer = ""

        return remaining


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class AIRouter:
    """Routes all interview generation strictly through Groq."""

    def __init__(self):
        self._setup_groq()

    def _setup_groq(self):
        api_key = os.getenv("GROQ_API_KEY")

        if api_key and api_key != "your_groq_api_key_here":
            self.system_groq_client = Groq(
                api_key=api_key
            )
            self.system_groq_available = True
        else:
            self.system_groq_client = None
            self.system_groq_available = False

            print(
                "[WARNING] AIRouter: GROQ_API_KEY is missing!"
            )

    def generate_answer(
        self,
        question: str,
        model: str = "groq",
        api_key: str = None,
        language: str = "python",
        system_prompt: str = None,
    ) -> str:
        """Generate a clean non-streaming answer."""

        return self._generate_with_groq(
            question,
            api_key,
            model,
            language,
            system_prompt,
        )

    def generate_answer_stream(
        self,
        question: str,
        model: str = "groq",
        api_key: str = None,
        language: str = "python",
        system_prompt: str = None,
    ) -> Generator[str, None, None]:
        """Generate a clean streaming answer."""

        yield from self._generate_with_groq_stream(
            question,
            api_key,
            model,
            language,
            system_prompt,
        )

    def generate_answer_from_image(
        self,
        image_data: str,
        prompt: str,
        api_key: str = None,
        model_id: str = "groq",
    ) -> str:
        """Generate an answer from an image using Groq Vision."""

        if "base64," in image_data:
            image_data = image_data.split(
                "base64,",
                1,
            )[1]

        client = None

        if api_key:
            client = Groq(api_key=api_key)
        elif self.system_groq_available:
            client = self.system_groq_client

        if not client:
            return "Groq API key not configured."

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/png;base64,"
                                        f"{image_data}"
                                    )
                                },
                            },
                        ],
                    }
                ],
                model=VISION_MODEL,
                temperature=0.3,
                max_completion_tokens=1024,
            )

            return clean_script_output(
                chat_completion.choices[0]
                .message
                .content
            )

        except Exception as e:
            return f"Groq Vision Error: {str(e)}"

    def generate_answer_from_image_stream(
        self,
        image_data: str,
        prompt: str,
        api_key: str = None,
        model_id: str = "groq",
    ) -> Generator[str, None, None]:
        result = self.generate_answer_from_image(
            image_data,
            prompt,
            api_key,
            model_id,
        )

        yield result

    def _get_client(
        self,
        api_key: str = None,
    ):
        if api_key:
            return Groq(api_key=api_key)

        if self.system_groq_available:
            return self.system_groq_client

        return None

    def _generate_with_groq(
        self,
        question: str,
        api_key: str = None,
        model_id: str = "llama-3.3-70b-versatile",
        language: str = "python",
        system_prompt: str = None,
    ) -> str:
        """Generate a final answer without exposing reasoning."""

        client = self._get_client(api_key)

        if not client:
            return "Groq API key not configured."

        groq_model = (
            model_id
            if model_id and model_id != "groq"
            else DEFAULT_MODEL
        )

        try:
            if system_prompt:
                sys_p = system_prompt
                user_p = question
            else:
                sys_p, user_p = build_interview_prompt(
                    question,
                    language,
                )

            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": sys_p,
                    },
                    {
                        "role": "user",
                        "content": user_p,
                    },
                ],
                model=groq_model,

                # CRITICAL:
                # Qwen reasoning is not returned to the UI.
                reasoning_format="hidden",

                # CRITICAL:
                # Interview answers do not need visible/deep reasoning.
                reasoning_effort="none",

                temperature=0.7,
                max_completion_tokens=700,
                stream=False,
            )

            raw_answer = (
                completion
                .choices[0]
                .message
                .content
            )

            return clean_script_output(
                raw_answer
            )

        except Exception as e:
            return f"Groq Error: {str(e)}"

    def _generate_with_groq_stream(
        self,
        question: str,
        api_key: str = None,
        model_id: str = "llama-3.3-70b-versatile",
        language: str = "python",
        system_prompt: str = None,
    ) -> Generator[str, None, None]:
        """
        Stream only final answer tokens.

        The critical protection is reasoning_format="hidden" plus
        reasoning_effort="none".
        """

        client = self._get_client(api_key)

        if not client:
            yield "Groq API key not configured."
            return

        groq_model = (
            model_id
            if model_id and model_id != "groq"
            else DEFAULT_MODEL
        )

        sanitizer = StreamSanitizer()

        try:
            if system_prompt:
                sys_p = system_prompt
                user_p = question
            else:
                sys_p, user_p = build_interview_prompt(
                    question,
                    language,
                )

            stream = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": sys_p,
                    },
                    {
                        "role": "user",
                        "content": user_p,
                    },
                ],
                model=groq_model,

                # CRITICAL:
                # Never return Qwen's <think> reasoning.
                reasoning_format="hidden",

                # CRITICAL:
                # Use non-thinking mode for fast interview dialogue.
                reasoning_effort="none",

                temperature=0.7,
                max_completion_tokens=700,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                content = getattr(
                    delta,
                    "content",
                    None,
                )

                if not content:
                    continue

                safe_content = sanitizer.feed(
                    content
                )

                if safe_content:
                    yield safe_content

            remaining = sanitizer.flush()

            if remaining:
                yield remaining

        except Exception as e:
            yield f"Groq Stream Error: {str(e)}"


# Singleton
ai_router = AIRouter()


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    model: str = "groq",
    api_key: str = None,
    language: str = "python",
) -> str:
    return ai_router.generate_answer(
        question,
        model,
        api_key,
        language,
    )


def generate_answer_stream(
    question: str,
    model: str = "groq",
    api_key: str = None,
    language: str = "python",
    system_prompt: str = None,
) -> Generator[str, None, None]:
    return ai_router.generate_answer_stream(
        question,
        model,
        api_key,
        language,
        system_prompt=system_prompt,
    )


def generate_answer_from_image(
    image_data: str,
    prompt: str,
    model: str = "groq",
    api_key: str = None,
) -> str:
    return ai_router.generate_answer_from_image(
        image_data,
        prompt,
        api_key=api_key,
        model_id=model,
    )


def generate_raw_prompt(
    prompt: str,
    model: str = "groq",
    api_key: str = None,
) -> str:
    """
    Raw Groq helper.

    Unlike interview generation, this is intentionally a generic endpoint,
    but it still hides reasoning so raw model analysis cannot leak.
    """

    client = ai_router._get_client(api_key)

    if not client:
        return "Groq NOT configured."

    try:
        model_to_use = (
            model
            if model and model != "groq"
            else DEFAULT_MODEL
        )

        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model_to_use,
            reasoning_format="hidden",
            reasoning_effort="none",
            temperature=0.7,
            max_completion_tokens=700,
        )

        return clean_script_output(
            completion.choices[0]
            .message
            .content
        )

    except Exception as e:
        return f"Groq Error: {str(e)}"
