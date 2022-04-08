import wave
import numpy as np
import matplotlib.pyplot as plt
import sys

from audio_read_test import waveform

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

###########################################################

file = "testRecord.wav"

waveform(file)
dump_audio(file)
