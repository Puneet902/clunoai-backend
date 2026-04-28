"""
Audio Diagnostic Script
Checks all audio devices and tests Stereo Mix capture.
"""

import sounddevice as sd
import numpy as np
import time


def list_all_devices():
    """List all audio devices with details."""
    print("\n" + "="*70)
    print("🔊 ALL AUDIO DEVICES ON YOUR SYSTEM")
    print("="*70)
    
    devices = sd.query_devices()
    
    print("\n📥 INPUT DEVICES (Recording):")
    print("-"*70)
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            default_marker = " ⭐ DEFAULT" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default_marker}")
            print(f"      Channels: {dev['max_input_channels']}, Sample Rate: {int(dev['default_samplerate'])} Hz")
            print()
    
    print("\n📤 OUTPUT DEVICES (Playback):")
    print("-"*70)
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            default_marker = " ⭐ DEFAULT" if i == sd.default.device[1] else ""
            print(f"  [{i}] {dev['name']}{default_marker}")
            print(f"      Channels: {dev['max_output_channels']}, Sample Rate: {int(dev['default_samplerate'])} Hz")
            print()


def find_stereo_mix():
    """Find Stereo Mix or loopback device."""
    print("\n" + "="*70)
    print("🔍 SEARCHING FOR STEREO MIX / LOOPBACK DEVICE")
    print("="*70)
    
    devices = sd.query_devices()
    stereo_mix = None
    
    for i, dev in enumerate(devices):
        dev_name = dev['name'].lower()
        if dev['max_input_channels'] > 0:
            if 'stereo mix' in dev_name or 'loopback' in dev_name or 'what u hear' in dev_name:
                stereo_mix = (i, dev)
                print(f"\n✅ FOUND: [{i}] {dev['name']}")
                print(f"   Sample Rate: {int(dev['default_samplerate'])} Hz")
                print(f"   Channels: {dev['max_input_channels']}")
                break
    
    if not stereo_mix:
        print("\n❌ NO STEREO MIX FOUND!")
        print("\n📌 How to enable Stereo Mix:")
        print("   1. Right-click speaker icon in taskbar → 'Sound settings'")
        print("   2. Scroll down → 'More sound settings'")
        print("   3. Go to 'Recording' tab")
        print("   4. Right-click empty space → 'Show Disabled Devices'")
        print("   5. Right-click 'Stereo Mix' → 'Enable'")
        print("   6. Right-click 'Stereo Mix' → 'Set as Default Device'")
    
    return stereo_mix


def test_audio_capture(device_index, duration=5):
    """Test audio capture from a device."""
    print("\n" + "="*70)
    print(f"🎤 TESTING AUDIO CAPTURE FOR {duration} SECONDS")
    print("="*70)
    print("\n⚠️  PLAY SOME AUDIO NOW! (YouTube, Music, Google Meet, etc.)")
    print("    The test will capture audio and show levels...\n")
    
    dev = sd.query_devices(device_index)
    sample_rate = int(dev['default_samplerate'])
    channels = min(dev['max_input_channels'], 2)
    
    audio_levels = []
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        audio_data = np.frombuffer(indata, dtype=np.int16)
        level = np.max(np.abs(audio_data))
        audio_levels.append(level)
        
        # Visual meter
        bar_length = min(50, int(level / 600))
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"\r  Level: [{bar}] {level:5d}", end="", flush=True)
    
    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=4096,
            dtype="int16",
            channels=channels,
            device=device_index,
            callback=callback
        ):
            time.sleep(duration)
        
        print("\n")
        
        if audio_levels:
            max_level = max(audio_levels)
            avg_level = sum(audio_levels) / len(audio_levels)
            
            print("\n📊 RESULTS:")
            print(f"   Max Level: {max_level}")
            print(f"   Avg Level: {avg_level:.0f}")
            
            if max_level < 100:
                print("\n❌ AUDIO LEVEL TOO LOW - NO AUDIO DETECTED!")
                print("\n📌 Possible causes:")
                print("   1. Stereo Mix is not capturing your audio output")
                print("   2. Your audio is playing through a DIFFERENT device")
                print("   3. Stereo Mix volume is set to 0")
                print("\n📌 Fix:")
                print("   1. Make sure audio plays through your default speakers")
                print("   2. Open Sound Settings → Recording → Stereo Mix → Properties → Levels → Set to 100")
            elif max_level < 1000:
                print("\n⚠️  AUDIO LEVEL LOW but detectable")
                print("   Try increasing Stereo Mix volume in Sound Settings")
            else:
                print("\n✅ AUDIO CAPTURE WORKING! Level is good.")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    print("\n" + "="*70)
    print("🔧 AUDIO DIAGNOSTIC TOOL")
    print("="*70)
    
    # List all devices
    list_all_devices()
    
    # Find Stereo Mix
    stereo_mix = find_stereo_mix()
    
    if stereo_mix:
        device_index, device = stereo_mix
        
        print("\n" + "="*70)
        input("Press ENTER to start audio capture test (play some audio first!)...")
        
        test_audio_capture(device_index)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
