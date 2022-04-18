import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import array
from pydub import AudioSegment
import audio_read_test as ar
import audio_isolation as iso
import sys
from scipy import signal as sig

# Converts a mono wav file to a stereo wave file with
# both channels having the same audio
def MonoToStereo(input, output):
    ifile = wave.open(input)
    print (ifile.getparams())

    # Get the audio parameters
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
    assert comptype == 'NONE'  # Compressed not supported yet

    # Get the audio data from the file and store in an array
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
    ifile.close()

    # Create the stereo array and populate
    stereo = 2 * left_channel
    stereo[0::2] = stereo[1::2] = left_channel

    # Create the stereo file, set the proper parameters, and write the data
    ofile = wave.open(output, 'w')
    ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(stereo.tobytes())
    ofile.close()

def AddDelay_mono(input, output, delay):
    # Create 1 sec of silence audio segment
    delay = AudioSegment.silent(duration=delay)  #duration in milliseconds

    # Read wav file to an audio segment
    rec = AudioSegment.from_wav(input)

    # Add above two audio segments
    rec = delay + rec

    # Export the modified file
    rec.export(output, format="wav")

# Combines two mono files together and adds padding
# to assure both are the proper length
def MonoToStereo_delayed(channel1, channel2, output):
    # Open files
    left_file = wave.open(channel1)
    right_file = wave.open(channel2)
    print (left_file.getparams())
    print (right_file.getparams())

    # Get the audio parameters
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = left_file.getparams()
    assert comptype == 'NONE'  # Compressed not supported yet
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = right_file.getparams()
    assert comptype == 'NONE'  # Compressed not supported yet

    # Get the audio data from the file and store in an array
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    left_channel = array.array(array_type, left_file.readframes(nframes))[::nchannels]
    right_channel = array.array(array_type, right_file.readframes(nframes))[::nchannels]
    left_file.close()
    right_file.close()

    # Create the stereo array and populate
    padding = [0] * (len(right_channel) - len(left_channel))
    left_channel.extend(padding)
    stereo = 2 * right_channel
    stereo[0::2] = left_channel
    stereo[1::2] = right_channel

    # Create the stereo file, set the proper parameters, and write the data
    ofile = wave.open(output, 'w')
    ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(stereo.tobytes())
    ofile.close()

# Currently only supports mono files
def addNoise(file, deviation):
    # Open the file and get the audio parameters
    rec = wave.open(file, "rb")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = rec.getparams()

    # Read the signal data and convert it to an integer array
    signal = rec.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')

    # Generate a noise array and add it to the original signal
    noise = np.random.normal(0, deviation, signal.shape).astype(dtype='int16')
    newSignal = np.add(signal, noise)

    # Create the stereo file, set the proper parameters, and write the data
    ofile = wave.open('noiseAdded.wav', 'wb')
    ofile.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(newSignal.tobytes())
    ofile.close()

def smooth_audio(file, window):
    # Open the file and get the audio parameters
    rec = wave.open(file, "rb")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = rec.getparams()

    # Read the signal data and convert it to an integer array
    signal = rec.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')

    # Make sure that the window size is odd
    if window % 2 == 0:
        window += 1

    # Create the output array and window_array
    smoothed = np.array([0] * (signal.size), dtype='int16')
    winArray = np.array(signal[0 : window])

    # Begin the smoothing process
    for i in range(signal.size - window - 1):
        winArray = np.roll(winArray, -1)
        winArray[window-1] = signal[i + window-1]
        smoothed[i] = np.mean(winArray)

    # Create the stereo file, set the proper parameters, and write the data
    ofile = wave.open('smoothed.wav', 'wb')
    ofile.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(smoothed.tobytes())
    ofile.close()


############################################################################################

if __name__ == '__main__':
    file = 'testRecord.wav'
    edit = 'noiseAdded.wav'
    trim = 'trimmedRecord.wav'
    smoothed = 'smoothed.wav'
    # MonoToStereo("testRecord.wav", "stereoRecord.wav")
    # AddDelay_mono("testRecord.wav", "delayedRecord.wav", 100)
    # MonoToStereo_delayed("test  Record.wav", "delayedRecord.wav", "timeShifted.wav")
    # ar.waveform(file)
    addNoise(file, 100)
    # iso.trim_audio(edit, 1000, 1000)
    # ar.waveform(edit)
    # iso.dump_audio(file)
    # iso.dump_audio(edit)
    smooth_audio(edit, 7)
    ar.waveform(edit)
    ar.waveform(smoothed)
    # iso.dump_audio(trim, 'trimDump.txt')
    # iso.dump_audio('smoothed.wav', 'smoothDump.txt')