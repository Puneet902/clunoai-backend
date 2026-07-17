"""
Speech to Text Module - Advanced Version
Uses OpenAI Whisper for transcription and WASAPI loopback for system audio capture.
Can capture audio from ANY output device including Bluetooth headphones.
"""

import os
import sys
import tempfile
import threading
import wave
import time
import json
import base64
from typing import Optional, Generator
from dotenv import load_dotenv

# Force UTF-8 output so emoji print() calls don't crash on Windows cp1252 console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

# Try to import pyaudiowpatch for WASAPI loopback (works with Bluetooth)
try:
    import pyaudiowpatch as pyaudio
    WASAPI_AVAILABLE = True
    print("[SUCCESS] PyAudioWPatch loaded - WASAPI loopback available (Bluetooth support)")
except ImportError:
    try:
        import pyaudio
        WASAPI_AVAILABLE = False
        print("[WARNING] Using standard PyAudio - Bluetooth capture may not work")
    except ImportError:
        pyaudio = None
        WASAPI_AVAILABLE = False
        print("[ERROR] PyAudio not available")

# Verify PyAudio can actually instantiate (fails on headless/cloud servers)
if pyaudio is not None:
    try:
        _pa_test = pyaudio.PyAudio()
        _pa_test.terminate()
    except Exception as _pa_err:
        print(f"⚠️ PyAudio module present but unusable (no audio devices): {_pa_err}")
        pyaudio = None
        WASAPI_AVAILABLE = False

# Try to import Groq for Whisper
try:
    from groq import Groq
    GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))
    if GROQ_AVAILABLE:
        print("[INFO] Groq Whisper API available (FREE & FAST!)")
except ImportError:
    GROQ_AVAILABLE = False

# Try to import Whisper (local or OpenAI API)
try:
    import openai
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here")
    if OPENAI_AVAILABLE:
        print("✅ OpenAI Whisper API available")
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import whisper
    WHISPER_LOCAL_AVAILABLE = True
    print("✅ Local Whisper model available")
except ImportError:
    WHISPER_LOCAL_AVAILABLE = False

# Fallback to Vosk if Whisper not available
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


class AdvancedSpeechToText:
    """
    Advanced speech-to-text using Whisper and WASAPI loopback.
    Supports capturing from Bluetooth and any audio device with Mixed-Mode.
    """
    
    def __init__(self):
        self.is_listening = False
        self.audio_frames = []
        self.stream = None
        self.mic_stream = None
        self.pa = None
        self.listen_thread = None
        self.loopback_device = None
        self.sample_rate = 16000 # Default target
        self.channels = 1
        self.text_buffer = ""
        self.whisper_model = None
        self.vosk_model = None
        self.recognizer = None
        self.transcription_method = None
        self._lock = threading.Lock()
        
        # Buffers for mixing
        self.system_queue = []
        self.mic_queue = []
        
        self.system_rate = 16000
        self.system_channels = 1
        
        self.mic_rate = 16000
        self.mic_channels = 1
        
        print("\n--- AI INTERVIEW ASSISTANT - OPTIMIZED BACKEND v2.1 ---")
        self._find_audio_device()
        self._init_models()
        print("[INFO] STT Engine: Ready")
    
    def _init_models(self):
        """Initialize transcription models."""
        if GROQ_AVAILABLE:
            self.transcription_method = "groq"
        elif WHISPER_LOCAL_AVAILABLE:
            self.transcription_method = "local"
        
        if VOSK_AVAILABLE:
            model_path = "vosk-model-small-en-us-0.15"
            if os.path.exists(model_path):
                try:
                    print(f"[INFO] Loading Vosk model from {model_path}...")
                    self.vosk_model = Model(model_path)
                    print("[SUCCESS] Vosk loaded for real-time feedback")
                except Exception as e:
                    print(f"[WARNING] Vosk load failed: {e}")

    def _find_audio_device(self):
        """Analyze and select audio devices with exhaustive logging."""
        if not pyaudio: return
        try:
            if self.pa is None:
                try:
                    self.pa = pyaudio.PyAudio()
                except Exception as pa_init_err:
                    print(f"[ERROR] PyAudio().init failed: {pa_init_err}")
                    return
            
            self.loopback_device = None
            self.mic_device = None
            
            print("\n[INFO] --- Audio Device Diagnostic ---")
            
            # Find WASAPI Host API index
            wasapi_api_index = -1
            try:
                for i in range(self.pa.get_host_api_count()):
                    api_info = self.pa.get_host_api_info_by_index(i)
                    if api_info.get('type') == pyaudio.paWASAPI:
                        wasapi_api_index = i
                        break
            except: pass

            for i in range(self.pa.get_device_count()):
                try:
                    dev = self.pa.get_device_info_by_index(i)
                    name = dev.get('name', '')
                    max_in = dev.get('maxInputChannels', 0)
                    is_loopback = dev.get('isLoopbackDevice', False)
                    rate = int(dev.get('defaultSampleRate', 0))
                    api_idx = dev.get('hostApi')
                    
                    print(f"Index {i}: {name} (In={max_in}, Loopback={is_loopback}, Rate={rate}Hz, API={api_idx})")
                    
                    # PRIORITY: WASAPI Loopback device
                    if is_loopback and (api_idx == wasapi_api_index or wasapi_api_index == -1):
                        # Use the FIRST loopback device we find as primary
                        if self.loopback_device is None:
                            self.loopback_device = i
                            self.system_rate = rate
                            self.system_channels = max_in
                            print(f"--- Found Loopback Device: {name}")

                    # FALLBACK: Microphone
                    if max_in > 0 and not is_loopback and self.mic_device is None:
                        # Prioritize devices with 'mic' or 'headset' in name
                        if any(x in name.lower() for x in ['mic', 'headset', 'usb']):
                            self.mic_device = i
                            self.mic_rate = rate
                            self.mic_channels = max_in

                except: pass

            # Final Selection Logging
            if self.loopback_device is not None:
                print(f"[SUCCESS] Selected Loopback ID={self.loopback_device} ({self.system_rate}Hz)")
            else:
                print("[WARNING] No Loopback device found. Looking for Stereo Mix...")
                # Try to find Stereo Mix if no explicit loopback
                for i in range(self.pa.get_device_count()):
                    try:
                        dev = self.pa.get_device_info_by_index(i)
                        if "stereo mix" in dev.get('name', '').lower():
                            self.loopback_device = i
                            self.system_rate = int(dev.get('defaultSampleRate'))
                            self.system_channels = dev.get('maxInputChannels')
                            print(f"[SUCCESS] Found Stereo Mix as Loopback: ID={i}")
                            break
                    except: pass

            if self.loopback_device is None:
                print("[WARNING] Falling back to default microphone.")
                try:
                    default_in = self.pa.get_default_input_device_info()
                    self.mic_device = default_in.get('index')
                    self.mic_rate = int(default_in.get('defaultSampleRate', 16000))
                    self.mic_channels = int(default_in.get('maxInputChannels', 1))
                except:
                    print("[ERROR] No audio input devices found.")
            print("-----------------------------------\n")
            
        except Exception as e:
            print(f"[ERROR] _find_audio_device crash: {e}")
            print("-----------------------------------\n")
            
        except Exception as e:
            print(f"[ERROR] Device search error: {e}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """System audio callback."""
        if self.is_listening:
            with self._lock: self.system_queue.append(in_data)
        return (in_data, pyaudio.paContinue)

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Mic audio callback."""
        if self.is_listening:
            with self._lock: self.mic_queue.append(in_data)
        return (in_data, pyaudio.paContinue)

    def _record_audio(self):
        """Mixer thread: Mix System + Mic and feed to engines."""
        import traceback
        import numpy as np
        from scipy import signal
        
        try:
            print(f"[INFO] Mixer started ({'System' if self.stream else 'OFF'} + {'Mic' if self.mic_stream else 'OFF'})")
            
            leftover_sys = np.array([], dtype=np.float32)
            leftover_mic = np.array([], dtype=np.float32)
            
            while self.is_listening:
                sys_frame = None
                mic_frame = None
                
                with self._lock:
                    if self.system_queue: sys_frame = self.system_queue.pop(0)
                    if self.mic_queue: mic_frame = self.mic_queue.pop(0)
                
                if not sys_frame and not mic_frame:
                    time.sleep(0.01)
                    continue
                
                if sys_frame:
                    arr = np.frombuffer(sys_frame, dtype=np.int16).astype(np.float32)
                    if self.system_channels > 1:
                        arr = arr.reshape(-1, self.system_channels).mean(axis=1)
                    if self.system_rate != 16000:
                        # Faster, safer linear interpolation for STT resampling
                        x = np.linspace(0, 1, len(arr))
                        x_new = np.linspace(0, 1, int(len(arr) * 16000 / self.system_rate))
                        arr = np.interp(x_new, x, arr)
                    leftover_sys = np.concatenate([leftover_sys, arr])
                
                if mic_frame:
                    arr = np.frombuffer(mic_frame, dtype=np.int16).astype(np.float32)
                    if self.mic_channels > 1:
                        arr = arr.reshape(-1, self.mic_channels).mean(axis=1)
                    if self.mic_rate != 16000:
                        x = np.linspace(0, 1, len(arr))
                        x_new = np.linspace(0, 1, int(len(arr) * 16000 / self.mic_rate))
                        arr = np.interp(x_new, x, arr)
                    leftover_mic = np.concatenate([leftover_mic, arr])
                
                # Mixing logic
                if self.mic_stream and self.stream:
                    available = min(len(leftover_sys), len(leftover_mic))
                    if available == 0:
                        # Asymmetric handling to prevent blocking - reduced threshold to 0.25s
                        if len(leftover_sys) > 4000:
                            self._store_and_feedback(leftover_sys[:4000], False)
                            leftover_sys = leftover_sys[4000:]
                        elif len(leftover_mic) > 4000:
                            self._store_and_feedback(leftover_mic[:4000], True)
                            leftover_mic = leftover_mic[4000:]
                        continue
                    else:
                        # Monitor levels occasionally
                        if time.time() % 5 < 0.02:
                            sys_p = np.abs(leftover_sys[:available]).max()
                            mic_p = np.abs(leftover_mic[:available]).max()
                            print(f"[DEBUG] Audio Sync: Sys Peak={sys_p:.0f}, Mic Peak={mic_p:.0f}, Buffers: S={len(leftover_sys)} M={len(leftover_mic)}")
                            
                        mixed = leftover_sys[:available] + leftover_mic[:available]
                        self._store_and_feedback(mixed, len(leftover_mic) >= len(leftover_sys), leftover_mic[:available])
                        leftover_sys = leftover_sys[available:]
                        leftover_mic = leftover_mic[available:]
                else:
                    active = leftover_sys if self.stream else leftover_mic
                    if len(active) > 0:
                        self._store_and_feedback(active, self.mic_stream is not None)
                        if self.stream: leftover_sys = np.array([], dtype=np.float32)
                        else: leftover_mic = np.array([], dtype=np.float32)
        except Exception:
            print("[ERROR] Mixer thread CRASHED:")
            traceback.print_exc()
        finally:
            print("[INFO] Mixer thread stopped")

    def _store_and_feedback(self, data_float32, source_is_mic=True, data_for_feedback=None):
        """Store audio and update feedback."""
        import numpy as np
        
        # Apply gain boost to final storage as well if signal is weak
        final_data = data_float32.copy()
        peak = np.abs(final_data).max()
        if 0 < peak < 10000:
            boost = min(15000.0 / peak, 5.0) # slightly more conservative for final
            final_data *= boost
            
        mixed_int16 = np.clip(final_data, -32768, 32767).astype(np.int16)
        with self._lock:
            self.audio_frames.append(mixed_int16.tobytes())
            
        fb_data = data_for_feedback if data_for_feedback is not None else data_float32
        self._update_vosk(fb_data)

    def _update_vosk(self, data_float32):
        """Update Vosk with gain boost and text accumulation."""
        if not getattr(self, 'recognizer', None): return
        import numpy as np
        peak = np.abs(data_float32).max()
        if peak > 0 and peak < 10000:
            boost = min(15000.0 / peak, 8.0)
            data_float32 = data_float32 * boost
        fb_int16 = np.clip(data_float32, -32768, 32767).astype(np.int16)
        try:
            if self.recognizer.AcceptWaveform(fb_int16.tobytes()):
                res = json.loads(self.recognizer.Result())
                if res.get("text"):
                    with self._lock:
                        # Append to accumulated text
                        if not hasattr(self, 'accumulated_text'): self.accumulated_text = ""
                        self.accumulated_text += res["text"] + " "
                        self.text_buffer = ""
            else:
                partial = json.loads(self.recognizer.PartialResult())
                if partial.get("partial"):
                    p = partial["partial"].strip()
                    if p: self.text_buffer = p
        except: pass

    def get_current_transcript(self, api_key: str = None) -> str:
        """Transcribe recent buffer without stopping the stream (lasts ~10 mins max)."""
        if not pyaudio: return ""
        with self._lock:
            if not self.audio_frames: return ""
            max_bytes = 16000 * 1 * 2 * 600  # 10 min cap
            frames_to_use = list(self.audio_frames)
            self.audio_frames = []          # reset so next click = fresh question
            self.accumulated_text = ""      # also clear Vosk accumulated text
            self.text_buffer = ""

            raw_audio = b''.join(frames_to_use)
            if len(raw_audio) > max_bytes:
                raw_audio = raw_audio[-max_bytes:]
            
        # --- NORMALIZATION ---
        try:
            import numpy as np
            samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            peak = np.abs(samples).max()
            if peak > 0 and peak < 12000:
                gain = min(20000.0 / peak, 8.0)
                samples = (samples * gain).clip(-32768, 32767)
                raw_audio = samples.astype(np.int16).tobytes()
        except: pass
        
        import io
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_audio)
        
        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"
        
        text = self._transcribe(wav_buffer, api_key)
        return text

    def start_listening(self) -> dict:
        """Start dual capture."""
        if self.is_listening: return {"success": False, "message": "Already listening"}

        # Cloud / Railway mode — no audio hardware available on server
        # Desktop captures audio client-side and sends via /transcribe-and-stream
        if not pyaudio:
            self.is_listening = True
            self.audio_frames = []
            self.text_buffer = "Cloud mode - client captures audio"
            print("[INFO] Cloud mode: no PyAudio, client-side capture in use")
            return {"success": True, "message": "Cloud mode - client handles audio capture"}

        self.is_listening = True
        self.audio_frames = []
        self.system_queue = []
        self.mic_queue = []
        self.text_buffer = "Listening..."

        try:
            if self.pa is None:
                try:
                    self.pa = pyaudio.PyAudio()
                except Exception as pa_err:
                    print(f"[ERROR] Cannot init PyAudio (no audio HW): {pa_err} — switching to cloud mode")
                    self.text_buffer = "Cloud mode - client captures audio"
                    return {"success": True, "message": "Cloud mode - client handles audio capture"}
            source = os.getenv("AUDIO_SOURCE", "auto").lower()
            
            # Start System stream
            self.stream = None
            if source in ["auto", "system"] and self.loopback_device is not None:
                print(f"[INFO] Opening Loopback ({self.system_rate}Hz, {self.system_channels}ch)")
                self.stream = self.pa.open(
                    format=pyaudio.paInt16, channels=self.system_channels,
                    rate=self.system_rate, input=True, input_device_index=self.loopback_device,
                    frames_per_buffer=int(self.system_rate * 0.1), stream_callback=self._audio_callback
                )
            
            # Start Mic stream
            self.mic_stream = None
            if source in ["auto", "microphone"] or self.loopback_device is None:
                # Use mic_device if explicitly found, else default
                dev_idx = self.mic_device
                dev_name = "Default"
                if dev_idx is not None:
                    try: dev_name = self.pa.get_device_info_by_index(dev_idx)['name']
                    except: pass
                
                print(f"[INFO] Opening Mic: {dev_name} (ID={dev_idx}, {self.mic_rate}Hz, {self.mic_channels}ch)")
                try:
                    self.mic_stream = self.pa.open(
                        format=pyaudio.paInt16, channels=self.mic_channels, rate=self.mic_rate,
                        input=True, input_device_index=dev_idx,
                        frames_per_buffer=int(self.mic_rate * 0.1), stream_callback=self._mic_callback
                    )
                except Exception as me:
                    if "-9997" in str(me) or "sample rate" in str(me).lower():
                        print(f"[WARNING] Mic failed at {self.mic_rate}Hz, retrying NATIVE settings...")
                        m_info = self.pa.get_device_info_by_index(dev_idx) if dev_idx is not None else self.pa.get_default_input_device_info()
                        self.mic_rate = int(m_info['defaultSampleRate'])
                        self.mic_channels = int(m_info['maxInputChannels'])
                        print(f"[INFO] Retrying Mic: {self.mic_rate}Hz, {self.mic_channels}ch")
                        self.mic_stream = self.pa.open(
                            format=pyaudio.paInt16, channels=self.mic_channels, rate=self.mic_rate,
                            input=True, input_device_index=dev_idx,
                            frames_per_buffer=int(self.mic_rate * 0.1), stream_callback=self._mic_callback
                        )
                    else: raise me
            
            if not self.stream and not self.mic_stream:
                self.is_listening = False
                return {"success": False, "message": "No audio devices found"}

            if self.stream: self.stream.start_stream()
            if self.mic_stream: self.mic_stream.start_stream()
            
            if VOSK_AVAILABLE and self.vosk_model:
                self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
            
            self.listen_thread = threading.Thread(target=self._record_audio)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            
            return {"success": True, "message": f"Listening started ({'Mixed' if self.stream and self.mic_stream else 'Single'} Mode)"}
        except Exception as e:
            self.is_listening = False
            print(f"[ERROR] Start error: {e}")
            return {"success": False, "message": str(e)}

    def stop_listening_and_get_text(self, api_key: str = None) -> dict:
        """Stop and transcribe."""
        if not self.is_listening: return {"success": False, "message": "Not listening", "text": ""}
        if not pyaudio:
            self.is_listening = False
            return {"success": True, "message": "Cloud mode", "text": ""}
        self.is_listening = False
        
        if self.stream:
            try: self.stream.stop_stream(); self.stream.close()
            except: pass
            self.stream = None
        if self.mic_stream:
            try: self.mic_stream.stop_stream(); self.mic_stream.close()
            except: pass
            self.mic_stream = None
            
        if self.listen_thread: self.listen_thread.join(timeout=1.0)
        
        with self._lock:
            if not self.audio_frames: return {"success": False, "message": "No audio captured.", "text": ""}
            raw_audio = b''.join(self.audio_frames)
            self.audio_frames = []
            
            # --- NORMALIZATION ---
            try:
                import numpy as np
                samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
                peak = np.abs(samples).max()
                if peak > 0 and peak < 12000:
                    gain = min(20000.0 / peak, 8.0)
                    print(f"[INFO] Boosting final transcription: {gain:.1f}x")
                    samples = (samples * gain).clip(-32768, 32767)
                    raw_audio = samples.astype(np.int16).tobytes()
            except: pass
            
            import io
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(raw_audio)
            
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"
            
            text = self._transcribe(wav_buffer, api_key)
            raw_text = get_current_transcript()
            print(f"[INFO] [LATENCY] Backend Transcription: '{raw_text[:60]}...'")
            self.text_buffer = text
            print(f"[INFO] Final Transcribed: {text}")
            return {"success": True, "message": "Done", "text": text}

    def _transcribe(self, audio_file, api_key: str = None) -> str:
        if api_key: return self._transcribe_groq(audio_file, api_key)
        if self.transcription_method == "groq": return self._transcribe_groq(audio_file)
        elif self.transcription_method == "openai": return self._transcribe_openai(audio_file)
        elif self.transcription_method == "local": return self._transcribe_whisper_local(audio_file)
        return ""

    def _transcribe_groq(self, audio_file, api_key: str = None) -> str:
        try:
            client = Groq(api_key=api_key) if api_key else Groq()
            # Use requested model or default to whisper-large-v3
            model = os.getenv("WHISPER_MODEL", "whisper-large-v3")
            print(f"[INFO] Using Whisper Model: {model}")
            # Use a prompt to help the model with technical interview context and grammar
            stt_prompt = "Technical job interview. Precise transcription of algorithms, system design, and coding concepts. Handle accents and minor speech errors gracefully while maintaining technical accuracy."
            transcript = client.audio.transcriptions.create(
                model=model,
                file=("audio.wav", audio_file.read()),
                language="en",
                prompt=stt_prompt
            )
            return transcript.text
        except Exception as e:
            print(f"[ERROR] Groq error: {e}")
            return ""

    def _transcribe_openai(self, audio_file) -> str:
        try:
            client = openai.OpenAI()
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", audio_file.read()),
                language="en"
            )
            return transcript.text
        except Exception as e:
            print(f"[ERROR] OpenAI error: {e}")
            return ""

    def _transcribe_whisper_local(self, audio_file) -> str:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_file.getvalue())
                temp_path = temp_file.name
            result = self.whisper_model.transcribe(temp_path, language="en")
            try: os.unlink(temp_path)
            except: pass
            return result["text"].strip()
        except Exception as e:
            print(f"[ERROR] Local Whisper error: {e}")
            return ""

    def get_current_text(self, max_words: int = 3000) -> str:
        accumulated = getattr(self, 'accumulated_text', "")
        full_text = (accumulated + self.text_buffer).strip()
        
        # Slicing the text to the last X words for latency/context optimization
        words = full_text.split()
        if len(words) > max_words:
            return "... " + " ".join(words[-max_words:])
        return full_text

    def reset_buffer(self):
        """Hard reset of audio logs and text accumulation."""
        with self._lock:
            self.audio_frames = []
            self.accumulated_text = ""
            self.text_buffer = ""
            # Re-init Vosk recognizer to clear partials
            if getattr(self, 'recognizer', None) and self.vosk_model:
                from vosk import KaldiRecognizer
                self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
            print("[INFO] Audio and Text buffers cleared")

    def is_model_loaded(self) -> bool:
        return self.transcription_method is not None


# Create singleton instance
stt_engine = AdvancedSpeechToText()

def start_listening() -> dict:
    return stt_engine.start_listening()

def stop_listening_and_get_text(api_key: str = None) -> dict:
    return stt_engine.stop_listening_and_get_text(api_key)

def get_current_transcript(api_key: str = None) -> str:
    return stt_engine.get_current_transcript(api_key)

def get_current_text() -> str:
    return stt_engine.get_current_text()

def reset_buffer():
    return stt_engine.reset_buffer()

def is_model_loaded() -> bool:
    return stt_engine.is_model_loaded()
