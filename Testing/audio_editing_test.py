import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import array

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

############################################################################################

MonoToStereo("testRecord.wav", "modifiedRecord.wav")