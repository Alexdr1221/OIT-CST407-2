import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import array
from pydub import AudioSegment

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

############################################################################################

# MonoToStereo("testRecord.wav", "stereoRecord.wav")
AddDelay_mono("testRecord.wav", "delayedRecord.wav", 100)
MonoToStereo_delayed("testRecord.wav", "delayedRecord.wav", "timeShifted.wav")