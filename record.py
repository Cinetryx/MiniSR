import sys
import os
import wave
import pyaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np

from random import randint
from scipy.io import wavfile 

def setSound():    
    audioname = "bye/audio_file"

    if os.path.exists("data/audio/"+audioname+".wav"):
        rnd = randint(0, 100)
        audioname += str(rnd)

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 2
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(
        format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        frames_per_buffer=CHUNK)

    print("recording...")
    frames = [] 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open("data/audio/"+audioname+".wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    spf = wave.open("data/audio/"+audioname+".wav", "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        

    plt.figure()
    plt.title("Signal Wave...")
    plt.plot(signal)
    plt.savefig("data/graph/"+audioname+".png")
    #plt.show()

#setSound()

def setTestSound():
    audioname = "merci2"

    if os.path.exists("data/test/"+audioname+".wav"):
        rnd = randint(0, 100)
        audioname += str(rnd)

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 2
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(
        format=FORMAT, channels=CHANNELS,
        rate=RATE, input=True,
        frames_per_buffer=CHUNK)

    print("recording...")
    frames = [] 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open("data/test/"+audioname+".wav", 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

setTestSound()