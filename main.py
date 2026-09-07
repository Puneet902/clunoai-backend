"""
AI Interview Assistant - FastAPI Backend
Real-time speech-to-text with AI answer generation.

Updated:
- Qwen 3.6 reasoning is disabled/hidden for interview output.
- Streaming path no longer exposes <think> / reasoning content.
- Technical answers are no longer forced into first-person language.
- Fixed resume/JD base64 field typo.
"""

import os
import sys
import re
import json
import base64
import io
import wave
import tempfile
import numpy as np
import PyPDF2

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from groq import Groq

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speech_to_text import (
    start_listening,
    stop_listening_and_get_text,
    get_current_text,
    is_model_loaded,
)
from ai_router import (
    generate_answer_stream,
    FAST_MODEL,
    VERSATILE_MODEL,
)
from prompt_builder import (
    set_resume,
    set_jd,
    get_resume,
    get_jd,
    clear_context,
)

# Load environment variables
load_dotenv(override=True)

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


app = FastAPI(
    title="AI Interview Assistant",
    description="Optimized Real-time AI Assistant",
    version="2.2.0",
)


# ---------------------------------------------------------------------------
# Validation error handling
# ---------------------------------------------------------------------------

from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"[ERROR] [422] Validation Failure: {exc.errors()}")
    print(f"[DEBUG] [422] Request Body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StartListenResponse(BaseModel):
    success: bool
    message: str


class StopListenRequest(BaseModel):
    model: Optional[str] = "groq"
    language: Optional[str] = "python"


class StopListenResponse(BaseModel):
    success: bool
    message: str
    question: str
    answer: str


class StatusResponse(BaseModel):
    api_status: str
    model_loaded: bool
    groq_configured: bool
    current_text: str
    has_resume: bool
    has_jd: bool


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    question: Optional[str] = None
    text: Optional[str] = None
    model: Optional[str] = "groq"
    language: Optional[str] = "python"


class GenerateResponse(BaseModel):
    success: bool
    question: str
    answer: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    text_preview: str


class ResumeUploadRequest(BaseModel):
    filename: str
    content_base_64: Optional[str] = None
    content_base64: Optional[str] = None


class ScreenAnalyzeRequest(BaseModel):
    image_data: Optional[str] = ""
    image_data_list: Optional[List[str]] = []
    extracted_text: Optional[str] = ""
    model: Optional[str] = "groq"
    language: Optional[str] = "python"


class ScreenAnalyzeResponse(BaseModel):
    success: bool
    extracted_text: str
    is_coding_question: bool
    answer: str
    code_blocks: List


class TranscribeAudioRequest(BaseModel):
    audio_base_64: Optional[str] = None
    audio_base64: Optional[str] = None
    model: Optional[str] = "groq"
    language: Optional[str] = "python"
    mode: Optional[str] = "full"


class BackendAudioAnalyzeRequest(BaseModel):
    model: Optional[str] = "groq"
    language: Optional[str] = "python"


# ---------------------------------------------------------------------------
# Question cleanup
# ---------------------------------------------------------------------------

def clean_extracted_question(extracted_text: str, raw_text: str) -> str:
    if not extracted_text:
        return raw_text

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        extracted_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    meta_prefixes = [
        r'^(the\s+)?(interview(er)?\s+)?(is\s+)?(asking|question|transcript)\s*(is)?:?\s*',
        r'^(extracted\s+)?question:\s*',
        r'^(here\s+is\s+the\s+question:?\s*)',
        r'^["\']|["\']$',
    ]

    for prefix in meta_prefixes:
        cleaned = re.sub(
            prefix,
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

    extracted_lower = cleaned.lower()

    refusal_keywords = [
        "no question",
        "cannot extract",
        "can't extract",
        "does not contain",
        "isn't a question",
        "without the rest of the transcript",
        "provided transcript does not",
        "cannot identify",
        "no clear question",
        "is not a question",
    ]

    if any(keyword in extracted_lower for keyword in refusal_keywords):
        return raw_text

    if len(cleaned) < 5:
        return raw_text

    return cleaned


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def sse(event_type: str, content: str) -> str:
    """Create a valid Server-Sent Events message."""
    return f"data: {json.dumps({'type': event_type, 'content': content}, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/start-listen", response_model=StartListenResponse)
async def api_start_listen():
    return start_listening()


@app.post("/stop-listen", response_model=StopListenResponse)
async def api_stop_listen(request: StopListenRequest):
    res = stop_listening_and_get_text()

    return StopListenResponse(
        success=res.get("success", False),
        message=res.get("message", ""),
        question="",
        answer="",
    )


@app.post("/analyze-backend-audio-stream")
async def api_analyze_backend_audio_stream(
    request: BackendAudioAnalyzeRequest,
):
    """
    Analyze audio recorded by the backend and stream the final interview answer.
    """

    async def event_generator():
        try:
            from speech_to_text import get_current_transcript, reset_buffer

            raw_text = get_current_transcript()
            reset_buffer()

            print(
                f"[INFO] [LATENCY] Backend Transcription: "
                f"'{raw_text[:100]}...'"
            )

            if not raw_text or len(raw_text) < 5:
                yield sse("error", "No speech detected")
                return

            question_text = clean_extracted_question(raw_text, raw_text)

            print(
                f"[LATENCY] Direct Fast-Path Question: "
                f"'{question_text[:100]}...'"
            )

            yield sse("question", question_text)

            for chunk in generate_answer_stream(
                question_text,
                request.model,
                None,
                request.language,
            ):
                if chunk:
                    yield sse("answer_chunk", chunk)

        except Exception as e:
            import traceback

            print(f"[ERROR] Backend Analyze error: {e}")
            traceback.print_exc()

            yield sse("error", str(e))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/transcribe-and-stream")
async def api_transcribe_and_stream(
    request: TranscribeAudioRequest,
):
    """
    Transcribe audio with Groq Whisper and immediately stream
    a clean interview answer.
    """

    async def event_generator():
        try:
            # 1. Decode base64 audio
            b64_data = request.audio_base_64 or request.audio_base64

            if not b64_data:
                yield sse("error", "No audio provided")
                return

            if "," in b64_data[:100]:
                b64_data = b64_data.split(",", 1)[1]

            try:
                audio_bytes = base64.b64decode(b64_data)
            except Exception:
                yield sse("error", "Invalid base64 audio")
                return

            if len(audio_bytes) < 500:
                yield sse("error", "Audio too short")
                return

            # 2. Detect audio format
            is_wav = audio_bytes[:4] == b"RIFF"

            if is_wav:
                try:
                    with io.BytesIO(audio_bytes) as audio_io:
                        with wave.open(audio_io, "rb") as wf_in:
                            params = wf_in.getparams()
                            frames = wf_in.readframes(wf_in.getnframes())

                    samples = np.frombuffer(
                        frames,
                        dtype=np.int16,
                    ).astype(np.float32)

                    if len(samples) == 0:
                        yield sse("error", "No audio detected")
                        return

                    peak = np.abs(samples).max()

                    if peak < 5:
                        yield sse("error", "No audio detected")
                        return

                    if 0 < peak < 12000:
                        gain = min(25000.0 / peak, 12.0)
                        samples = (
                            samples * gain
                        ).clip(
                            -32768,
                            32767,
                        ).astype(np.int16)

                        frames = samples.tobytes()

                    out_io = io.BytesIO()

                    with wave.open(out_io, "wb") as wf_out:
                        wf_out.setparams(params)
                        wf_out.writeframes(frames)

                    out_io.seek(0)

                    audio_for_whisper = out_io.read()
                    audio_filename = "audio.wav"

                except Exception as ae:
                    print(f"[WARNING] WAV normalization failed: {ae}")
                    audio_for_whisper = audio_bytes
                    audio_filename = "audio.wav"

            else:
                # Browser MediaRecorder usually sends WebM/Opus.
                audio_for_whisper = audio_bytes
                audio_filename = "audio.webm"

                print(
                    f"[INFO] Received WebM audio "
                    f"({len(audio_bytes) // 1024}KB)"
                )

            # 3. Groq Whisper transcription
            groq_key = os.getenv("GROQ_API_KEY")

            if not groq_key:
                yield sse("error", "GROQ_API_KEY is not configured")
                return

            groq_client = Groq(api_key=groq_key)

            whisper_model = os.getenv(
                "WHISPER_MODEL",
                "whisper-large-v3",
            )

            print(
                f"[INFO] Processing {audio_filename} "
                f"({len(audio_for_whisper)} bytes)"
            )

            raw_text = ""

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=os.path.splitext(audio_filename)[1],
                    delete=False,
                ) as tmp:
                    tmp.write(audio_for_whisper)
                    tmp_path = tmp.name

                try:
                    with open(tmp_path, "rb") as f:
                        transcription = (
                            groq_client.audio.transcriptions.create(
                                model=whisper_model,
                                file=(audio_filename, f),
                                language="en",
                            )
                        )

                    raw_text = transcription.text.strip()

                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            except Exception as e:
                err_msg = str(e).lower()

                print(
                    f"[ERROR] Groq rejected {audio_filename}: {e}"
                )

                if "no audio track" in err_msg:
                    yield sse(
                        "error",
                        "No audio track found. Please ensure audio sharing is enabled.",
                    )
                    return

                # WebM fallback
                if not is_wav:
                    try:
                        print(
                            "[INFO] Trying audio.mp3 fallback..."
                        )

                        with tempfile.NamedTemporaryFile(
                            suffix=".mp3",
                            delete=False,
                        ) as tmp:
                            tmp.write(audio_for_whisper)
                            tmp_path = tmp.name

                        try:
                            with open(tmp_path, "rb") as f:
                                transcription = (
                                    groq_client.audio.transcriptions.create(
                                        model=whisper_model,
                                        file=("audio.mp3", f),
                                        language="en",
                                    )
                                )

                            raw_text = transcription.text.strip()

                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

                    except Exception:
                        raise e
                else:
                    raise

            print(
                f"[LATENCY] Transcription: "
                f"'{raw_text[:100]}...'"
            )

            if not raw_text or len(raw_text) < 5:
                yield sse(
                    "error",
                    "No speech detected in audio",
                )
                return

            # 4. Fast path: do not perform another LLM call just to
            # identify the question.
            question_text = clean_extracted_question(
                raw_text,
                raw_text,
            )

            print(
                f"[LATENCY] Direct Fast-Path Question: "
                f"'{question_text[:100]}...'"
            )

            yield sse("question", question_text)

            # 5. Stream only the final answer.
            for chunk in generate_answer_stream(
                question_text,
                request.model,
                None,
                request.language,
            ):
                if chunk:
                    yield sse("answer_chunk", chunk)

        except Exception as e:
            import traceback

            print(f"[ERROR] Transcribe error: {e}")
            traceback.print_exc()

            yield sse("error", str(e))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/generate")
async def api_generate_stream(
    request: GenerateRequest,
    x_api_key: Optional[str] = Header(
        None,
        alias="x-groq-api-key",
    ),
    x_generic_key: Optional[str] = Header(
        None,
        alias="x-api-key",
    ),
):
    """Generate an answer from manual text input."""

    api_key = x_generic_key or x_api_key
    prompt = request.prompt or request.question or request.text

    async def event_generator():
        try:
            if not prompt:
                yield sse("error", "No question provided")
                return

            for chunk in generate_answer_stream(
                prompt,
                request.model,
                api_key,
                request.language,
            ):
                if chunk:
                    yield sse("answer_chunk", chunk)

        except Exception as e:
            yield sse("error", str(e))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Resume / JD context
# ---------------------------------------------------------------------------

@app.post("/upload-resume-base64", response_model=UploadResponse)
async def upload_resume_base64(
    request: ResumeUploadRequest,
):
    """Upload a resume as Base64."""

    try:
        # FIXED: support both field names.
        b64_data = (
            request.content_base64
            or request.content_base_64
        )

        if not b64_data:
            raise ValueError("Resume content is missing")

        if "," in b64_data[:100]:
            b64_data = b64_data.split(",", 1)[1]

        content = base64.b64decode(b64_data)

        if request.filename.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(
                io.BytesIO(content)
            )

            text = "\n".join(
                p.extract_text() or ""
                for p in reader.pages
            ).strip()

        else:
            text = content.decode(
                "utf-8",
                errors="replace",
            )

        set_resume(text)

        return UploadResponse(
            success=True,
            message="Resume uploaded!",
            text_preview=text[:200],
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@app.post("/upload-jd-base64", response_model=UploadResponse)
async def upload_jd_base64(
    request: ResumeUploadRequest,
):
    """Upload a job description as Base64."""

    try:
        b64_data = (
            request.content_base64
            or request.content_base_64
        )

        if not b64_data:
            raise ValueError("JD content is missing")

        if "," in b64_data[:100]:
            b64_data = b64_data.split(",", 1)[1]

        content = base64.b64decode(b64_data)

        text = content.decode(
            "utf-8",
            errors="replace",
        )

        set_jd(text)

        return UploadResponse(
            success=True,
            message="JD uploaded!",
            text_preview=text[:100],
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@app.get("/status", response_model=StatusResponse)
async def api_status():
    return StatusResponse(
        api_status="online",
        model_loaded=is_model_loaded(),
        groq_configured=bool(
            os.getenv("GROQ_API_KEY")
        ),
        current_text=get_current_text(),
        has_resume=bool(get_resume()),
        has_jd=bool(get_jd()),
    )


@app.post("/clear-context")
async def api_clear_context():
    clear_context()
    return {"success": True}


@app.post("/analyze-screen-stream")
async def api_analyze_screen_stream(
    request: ScreenAnalyzeRequest,
):
    """
    Stream an answer for already-extracted screen text.

    Image OCR/extraction remains the responsibility of the caller,
    preserving the existing screen-analysis flow.
    """

    async def event_generator():
        text_context = request.extracted_text

        if not text_context:
            yield sse(
                "error",
                "No extracted screen text provided",
            )
            return

        yield sse(
            "question",
            text_context[:1000],
        )

        for chunk in generate_answer_stream(
            text_context,
            request.model,
            None,
            request.language,
        ):
            if chunk:
                yield sse("answer_chunk", chunk)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    print(
        "\n--- AI INTERVIEW ASSISTANT "
        "OPTIMIZED BACKEND v2.2 ---"
    )

    print(
        "GROQ_API_KEY:",
        "[READY]"
        if os.getenv("GROQ_API_KEY")
        else "[MISSING]",
    )

    print(
        "GROQ_MODEL:",
        os.getenv(
            "GROQ_MODEL",
            "qwen/qwen3.6-27b",
        ),
    )

    print(
        "WHISPER_MODEL:",
        os.getenv(
            "WHISPER_MODEL",
            "whisper-large-v3",
        ),
    )

    print(
        "Reasoning:",
        "DISABLED for interview responses",
    )

    print("=" * 60 + "\n")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
