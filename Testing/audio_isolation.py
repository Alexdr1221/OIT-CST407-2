import audio_read_test
import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io.wavfile import write

def dump_audio(file):
    # Open the input and output files
    rec = wave.open(file, "rb")
    dump = open("audio_dump.txt", "w")

    # Read the signal data and convert it to an integer array
    signal = rec.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    # Print the entire array
    np.set_printoptions(threshold=sys.maxsize)

    # Dump the values
    print("Dumping Sample Values...")
    dump.write(np.array2string(signal))
    print("Dumping Complete")
    dump.close()

def trim_audio(file, threshold):
    # Settings
    channels = 1
    sample_rate = 44100
    frame_size = 1024

    # Open the input and output files
    rec = wave.open(file, "rb")
    dump = open("audio_dump.txt", "w")

    # Read the signal data and convert it to an integer array
    signal = rec.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    firstval = np.argmax(signal > threshold)
    trimmed = signal[firstval:]

    # Save the audio as a .wav file
    trimmed = np.int16(trimmed/np.max(np.abs(trimmed)) * 32767)
    write("trimmedRecord.wav", 44100, trimmed)

###########################################################

file = "testRecord.wav"
trimmed = "trimmedRecord.wav"

trim_audio(file, 2000)
audio_read_test.waveform(trimmed)