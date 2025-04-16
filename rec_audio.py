import os
import pyaudio
import wave
from halo import Halo
import warnings

warnings.filterwarnings('ignore')

# import speech_recognition as sr

# for index, name in enumerate(sr.Microphone.list_microphone_names()):
#     print("Microphone with name \"{1}\" found for microphone(device_index{0})".format(index, name))


# First to use PyAudio to record customers call and save with Wave
# # Set params
chunk = 1024 # Recording in chunks
sam_format = pyaudio.paInt16 # 16 bits per sample
channels = 1
fs = 44100 # Record at 44100 (in kilohertze) samples/sec
seconds = 5
audio_file = "customer_call.wav"

# Create a portaudio interface 
p_audio = pyaudio.PyAudio()

with Halo(text='Recording', spinner='dots'):
    stream_audio = p_audio.open(format=sam_format,
                                channels=channels,
                                rate=fs,
                                frames_per_buffer=chunk,
                                input=True)

    # Intializing array to store frames
    audio_frames = []

    try:
        while True:
            audio_data = stream_audio.read(chunk)
            audio_frames.append(audio_data)
    except KeyboardInterrupt:
        
        pass
    
    # spinner.stop()
    stream_audio.stop_stream()
    stream_audio.close()
    p_audio.terminate() # Terminate the PortAudio interface

# Save the recording data as a WAV file 
wav_file = wave.open(audio_file, 'wb')
wav_file.setnchannels(channels)
wav_file.setsampwidth(p_audio.get_sample_size(sam_format))
wav_file.setframerate(fs)
wav_file.writeframes(b''.join(audio_frames))
wav_file.close()

