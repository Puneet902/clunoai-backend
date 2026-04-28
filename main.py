"""
AI Interview Assistant - FastAPI Backend
Real-time speech-to-text with AI answer generation.
Optimized for sub-second latency (<1s).
"""

import os
import sys
import json
import base64
import io
import wave
import tempfile
import numpy as np
import PyPDF2
from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from groq import Groq

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speech_to_text import start_listening, stop_listening_and_get_text, get_current_text, is_model_loaded
from ai_router import generate_answer, generate_answer_stream, generate_raw_prompt, FAST_MODEL, VERSATILE_MODEL
from prompt_builder import set_resume, set_jd, get_resume, get_jd, clear_context, clean_filler_words

# Load environment variables
load_dotenv(override=True)

# Create FastAPI app
app = FastAPI(
    title="AI Interview Assistant",
    description="Optimized Real-time AI Assistant",
    version="2.1.0"
)

# Debug: Catch and Print Pydantic Errors to find the 422 "Root Cause"
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"❌ [422 ERROR] Validation Failure: {exc.errors()}")
    print(f"🔍 [422 ERROR] Request Body: {await request.body()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---

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
    content_base64: Optional[str] = None # Support both naming conventions

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

# --- Optimized Endpoints ---

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
        answer=""
    )

@app.post("/analyze-backend-audio-stream")
async def api_analyze_backend_audio_stream(request: BackendAudioAnalyzeRequest):
    """
    Analyzes audio strictly recorded by the backend's PyAudioWPatch (fixes Bluetooth capture limitation).
    Target: <1s generation from audio buffer.
    """
    async def event_generator():
        try:
            from speech_to_text import get_current_transcript
            raw_text = get_current_transcript()
            print(f"📝 [LATENCY] Backend Transcription: '{raw_text[:60]}...'")

            if not raw_text or len(raw_text) < 5:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No speech detected'})}\n\n"
                return

            # FAST-PATH Extraction
            words = raw_text.split()
            fast_path_triggers = ["what", "how", "why", "define", "explain", "describe", "difference", "can you"]
            is_clear_question = raw_text.endswith('?') or any(raw_text.lower().startswith(t) for t in fast_path_triggers)
            
            if len(words) < 15 and is_clear_question:
                question_text = raw_text
                print("⚡ [LATENCY] Fast-Path Triggered: Skipping extraction LLM")
            else:
                extraction_prompt = f"Analyze the transcription and extract ONLY the precise question asked by the interviewer. Ignore the candidate's voice, background noise, fillers, or small talk. Return ONLY the clean question text.\n\nTRANSCRIPT: {raw_text}"
                question_text = generate_raw_prompt(extraction_prompt, model=FAST_MODEL).strip()
                print(f"🎯 [LATENCY] Extracted via {FAST_MODEL}: {question_text[:60]}...")
            
            yield f"data: {json.dumps({'type': 'question', 'content': question_text})}\n\n"
            
            for chunk in generate_answer_stream(question_text, request.model, None, request.language):
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk})}\n\n"
                
        except Exception as e:
            import traceback
            print(f"❌ Backend Analyze error: {e}")
            traceback.print_exc()
            try:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            except:
                pass
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/transcribe-and-stream")
async def api_transcribe_and_stream(request: TranscribeAudioRequest):
    """
    Ultra-low latency transcription and answer generation.
    Target: <1s from audio to first answer token.
    Uses Memory-only processing and Fast-Path bypass.
    """
    async def event_generator():
        try:
            # 1. Decode base64 audio (Memory-only)
            b64_data = request.audio_base_64 or request.audio_base64
            if not b64_data:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No audio provided'})}\n\n"
                return
            if "," in b64_data[:100]:
                b64_data = b64_data.split(",", 1)[1]
            audio_bytes = base64.b64decode(b64_data)
            if len(audio_bytes) < 500:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Audio too short'})}\n\n"
                return
            
            # 2. Fast Normalization (In-Memory)
            try:
                with io.BytesIO(audio_bytes) as audio_io:
                    with wave.open(audio_io, 'rb') as wf_in:
                        params = wf_in.getparams()
                        frames = wf_in.readframes(wf_in.getnframes())
                
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                peak = np.abs(samples).max()
                
                if peak < 5:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'No audio detected'})}\n\n"
                    return

                # Auto-gain if volume is low (< 40% of max)
                if 0 < peak < 12000:
                    gain = min(25000.0 / peak, 12.0)
                    samples = (samples * gain).clip(-32768, 32767).astype(np.int16)
                    frames = samples.tobytes()
                
                # Create memory buffer for Whisper
                out_io = io.BytesIO()
                with wave.open(out_io, 'wb') as wf_out:
                    wf_out.setparams(params)
                    wf_out.writeframes(frames)
                out_io.seek(0)
                audio_for_whisper = out_io.read()
            except Exception as ae:
                print(f"⚠️ Normalization failed: {ae}")
                audio_for_whisper = audio_bytes
            
            # 3. Transcribe with Groq Whisper
            groq_key = os.getenv("GROQ_API_KEY")
            client = Groq(api_key=groq_key)
            
            transcript = client.audio.transcriptions.create(
                model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
                file=("audio.wav", audio_for_whisper),
                language="en"
            )
            raw_text = transcript.text.strip()
            print(f"📝 [LATENCY] Transcription: '{raw_text[:60]}...'")

            if not raw_text or len(raw_text) < 5:
                return

            # 4. FAST-PATH: Bypass extraction LLM for short/clear questions
            question_text = raw_text
            words = raw_text.split()
            
            fast_path_triggers = ["what", "how", "why", "define", "explain", "describe", "difference", "can you"]
            is_clear_question = raw_text.endswith('?') or any(raw_text.lower().startswith(t) for t in fast_path_triggers)
            
            if len(words) < 15 and is_clear_question:
                print("⚡ [LATENCY] Fast-Path Triggered: Skipping extraction LLM")
            else:
                # 5. Fast Extraction (using Instant model)
                extraction_prompt = f"Analyze the transcription and extract ONLY the precise question asked by the interviewer. Ignore the candidate's voice, background noise, fillers, or small talk. Return ONLY the clean question text.\n\nTRANSCRIPT: {raw_text}"
                question_text = generate_raw_prompt(extraction_prompt, model=FAST_MODEL).strip()
                print(f"🎯 [LATENCY] Extracted via {FAST_MODEL}: {question_text[:60]}...")
            
            # Send question to client immediately
            yield f"data: {json.dumps({'type': 'question', 'content': question_text})}\n\n"
            
            # 6. Stream High-Quality Answer
            for chunk in generate_answer_stream(question_text, request.model, None, request.language):
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk})}\n\n"
                
        except Exception as e:
            import traceback
            print(f"❌ Transcribe error: {e}")
            traceback.print_exc()
            try:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            except:
                pass
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/generate")
async def api_generate_stream(
    request: GenerateRequest,
    x_api_key: Optional[str] = Header(None, alias="x-groq-api-key"),
    x_generic_key: Optional[str] = Header(None, alias="x-api-key")
):
    """Generate answer from manual text input (streaming)."""
    api_key = x_generic_key or x_api_key
    prompt = request.prompt or request.question # Fix 422: Handle both payload fields
    
    async def event_generator():
        try:
            if not prompt: return
            for chunk in generate_answer_stream(prompt, request.model, api_key, request.language):
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- Legacy & Context Endpoints ---

@app.post("/upload-resume-base64", response_model=UploadResponse)
async def upload_resume_base64(request: ResumeUploadRequest):
    """Upload resume as Base64."""
    try:
        b64_data = request.content_base64 or request.content_base_6_4
        if "," in b64_data[:100]: b64_data = b64_data.split(",", 1)[1]
        content = base64.b64decode(b64_data)
        
        if request.filename.lower().endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
        else:
            text = content.decode('utf-8')
            
        set_resume(text)
        return UploadResponse(success=True, message="Resume uploaded!", text_preview=text[:200])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-jd-base64", response_model=UploadResponse)
async def upload_jd_base64(request: ResumeUploadRequest):
    """Upload JD as Base64."""
    try:
        b64_data = request.content_base64 or request.content_base_6_4
        if "," in b64_data[:100]: b64_data = b64_data.split(",", 1)[1]
        content = base64.b64decode(b64_data)
        text = content.decode('utf-8')
        set_jd(text)
        return UploadResponse(success=True, message="JD uploaded!", text_preview=text[:100])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def api_status():
    return StatusResponse(
        api_status="online",
        model_loaded=is_model_loaded(),
        groq_configured=bool(os.getenv("GROQ_API_KEY")),
        current_text=get_current_text(),
        has_resume=bool(get_resume()),
        has_jd=bool(get_jd())
    )

@app.post("/clear-context")
async def api_clear_context():
    clear_context()
    return {"success": True}

@app.post("/analyze-screen-stream")
async def api_analyze_screen_stream(request: ScreenAnalyzeRequest):
    """Simple screen analysis stream."""
    async def event_generator():
        text_context = request.extracted_text
        if text_context:
            yield f"data: {json.dumps({'type': 'question', 'content': text_context[:100]})}\n\n"
            for chunk in generate_answer_stream(text_context, request.model):
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': chunk})}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Startup event
@app.on_event("startup")
async def startup_event():
    print("\n🚀 AI INTERVIEW ASSISTANT - OPTIMIZED BACKEND v2.1")
    print(f"GROQ_API_KEY: {'[READY]' if os.getenv('GROQ_API_KEY') else '[MISSING]'}")
    print(f"WHISPER_MODEL: {os.getenv('WHISPER_MODEL', 'whisper-large-v3')}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
