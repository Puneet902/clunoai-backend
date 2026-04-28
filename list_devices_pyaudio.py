import pyaudiowpatch as pyaudio
import os

def list_pyaudio_devices():
    pa = pyaudio.PyAudio()
    print("\n🔍 --- PyAudioWPatch Device List ---")
    
    wasapi_api_index = None
    for i in range(pa.get_host_api_count()):
        api = pa.get_host_api_info_by_index(i)
        if api.get('type') == pyaudio.paWASAPI:
            wasapi_api_index = i
            print(f"Found WASAPI API at index {i}")
            break
            
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        name = dev.get('name', '')
        max_in = dev.get('maxInputChannels', 0)
        is_loopback = dev.get('isLoopbackDevice', False)
        rate = int(dev.get('defaultSampleRate', 0))
        host_api = dev.get('hostApi')
        
        marker = ""
        if is_loopback: marker += " [LOOPBACK]"
        if max_in > 0 and not is_loopback: marker += " [INPUT/MIC]"
        if host_api == wasapi_api_index: marker += " (WASAPI)"
        
        print(f"Index {i}: {name}{marker}")
        print(f"    Channels: {max_in}, Rate: {rate}Hz")

    pa.terminate()

if __name__ == "__main__":
    list_pyaudio_devices()
