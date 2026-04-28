import pyaudiowpatch as pyaudio

def list_loopback_devices():
    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output_idx = wasapi_info["defaultOutputDevice"]
        default_output = p.get_device_info_by_index(default_output_idx)
        print(f"Default Output Device: {default_output['name']} (Index: {default_output_idx})")
        print("-" * 30)
        
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['hostApi'] == wasapi_info['index']:
                is_loopback = dev.get('isLoopbackDevice', False)
                if is_loopback:
                    print(f"LOOPBACK Device {i}: {dev['name']}")
                    print(f"  Matches Default? {'Yes' if default_output['name'] in dev['name'] else 'No'}")
                    print(f"  Max Input Channels: {dev['maxInputChannels']}")
                    print("-" * 30)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()

if __name__ == "__main__":
    list_loopback_devices()
