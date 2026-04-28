import os
import wave
import time
from speech_to_text import stt_engine

def test_recording():
    print("--- ClunoAI Audio Diagnostic ---")
    print(f"Current AUDIO_SOURCE: {os.getenv('AUDIO_SOURCE', 'system')}")
    
    # Check model
    if not stt_engine.is_model_loaded():
        print("❌ Transcription model not loaded.")
    else:
        print(f"✅ Transcription method: {stt_engine.transcription_method}")

    # Find device
    stt_engine._find_loopback_device()
    if stt_engine.loopback_device is not None:
        print(f"✅ Target Device: {stt_engine.loopback_device}")
    else:
        print("🎤 Target Device: Default Microphone")

    print("\n🎤 RECORDING TEST: Please speak for 5 seconds...")
    res = stt_engine.start_listening()
    if not res["success"]:
        print(f"❌ Failed to start: {res['message']}")
        return

    for i in range(5, 0, -1):
        print(f"Recording... {i}")
        time.sleep(1)

    print("\n⏹ STOPPING & TRANSCRIBING...")
    result = stt_engine.stop_listening_and_get_text()
    
    if result["success"]:
        print(f"\n✅ SUCCESS!")
        print(f"📝 Transcription: \"{result['text']}\"")
        if not result["text"]:
            print("⚠️ WARNING: Audio was captured but no text was generated. Check if your mic is muted.")
    else:
        print(f"\n❌ FAILED: {result['message']}")

if __name__ == "__main__":
    test_recording()
