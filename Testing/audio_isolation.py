import audio_read_test
import wave
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io.wavfile import write

def dump_audio(in_file, out_file):
    # Open the input and output files
    rec = wave.open(in_file, "rb")
    dump = open(out_file, "w")

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

def trim_audio(file, start_threshold, end_threshold):
    # Settings
    channels = 1
    sample_rate = 44100
    frame_size = 1024

    # Open the input file
    rec = wave.open(file, "rb")

    # Stereo Processing
    if rec.getnchannels() == 2:
        # Get the audio parameters
        (nchannels, sampwidth, framerate, nframes, comptype, compname) = rec.getparams()
        assert comptype == 'NONE'  # Compressed not supported yet

        # Read the signal data and convert it to an integer array
        signal = rec.readframes(-1)

        # Convert the signal array into raw sample values
        signal = np.frombuffer(signal, dtype="int16")

        # Split the signal into left and right channels
        left_channel = signal[0::2]
        right_channel = signal[1::2]

        # Locate the first and last index for each channel where the signal value is above the desired threshold
        left_first = min(idx for idx, val in enumerate(left_channel) if abs(val) > start_threshold)
        left_second = max(idx for idx, val in enumerate(left_channel) if abs(val) > end_threshold)
        right_first = min(idx for idx, val in enumerate(right_channel) if abs(val) > start_threshold)
        right_second = max(idx for idx, val in enumerate(right_channel) if abs(val) > end_threshold)

        # Determine the true start and end of the interesting part
        firstval = left_first if left_first < right_first else right_first
        secondval = left_second if left_second > right_second else right_second

        # Construct the trimmed arrays
        left_trim = left_channel[firstval:secondval]
        right_trim = right_channel[firstval:secondval]

        # Create the stereo array and populate
        stereo = 2 * right_trim.tolist()
        stereo[0::2] = left_trim
        stereo[1::2] = right_trim
        stereo = np.array(stereo)

        # Create the stereo file, set the proper parameters, and write the data
        ofile = wave.open("trimmedRecord.wav", 'w')
        ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
        ofile.writeframes(stereo.tobytes())
        ofile.close()
        return

    # Mono Processing
    else:
        # Read the signal data and convert it to an integer array
        signal = rec.readframes(-1)
        signal = np.frombuffer(signal, dtype="int16")

        # Locate the first and last index where the signal value is above the desired threshold
        firstval = min(idx for idx, val in enumerate(signal) if val > start_threshold)
        secondval = max(idx for idx, val in enumerate(signal) if val > end_threshold)

        # Construct the trimmed array
        trimmed = signal[firstval:secondval]

        # Save the audio as a .wav file
        trimmed = np.int16(trimmed/np.max(np.abs(trimmed)) * 32767)
        write("trimmedRecord.wav", 44100, trimmed)

###########################################################

if __name__ == '__main__':
    file = "testRecord.wav"
    trimmed = "trimmedRecord.wav"

    # audio_read_test.waveform(file)
    trim_audio(file, 6000, 1000)
    # audio_read_test.waveform(trimmed)
    dump_audio(trimmed, 'dump.txt')