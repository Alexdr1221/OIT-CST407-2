import pyaudio
import wave
import numpy as np

# Settings
CHUNK          = 44100  # frames to keep in buffer between reads
samp_rate      = 44100 # sample rate [Hz]
pyaudio_format = pyaudio.paInt16 # 16-bit device
buffer_format  = np.int16 # 16-bit for buffer
chans          = 2 # only read 1 channel
dev_index      = 1 # index of sound device

# Init Audio Object+
audio = pyaudio.PyAudio() # create pyaudio instantiation
##############################
### create pyaudio stream  ###
# -- streaming can be broken down as follows:
# -- -- format             = bit depth of audio recording (16-bit is standard)
# -- -- rate               = Sample Rate (44.1kHz, 48kHz, 96kHz)
# -- -- channels           = channels to read (1-2, typically)
# -- -- input_device_index = index of sound device
# -- -- input              = True (let pyaudio know you want input)
# -- -- frmaes_per_buffer  = chunk to grab and keep in buffer before reading
##############################
stream = audio.open(format = pyaudio_format, rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=CHUNK)

# Recorded audio stored as frames
frames = []

try:
    print("Recording...")
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

# CTRL C stops the recording
except KeyboardInterrupt:
    pass

# Stop the recording process
print("Ending Recording...")
stream.stop_stream()
stream.close()
audio.terminate()

# Save the audio as a .wav file
sound_file = wave.open("testRecord.wav", "wb")
sound_file.setnchannels(chans)
sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
sound_file.setframerate(samp_rate)
sound_file.writeframes(b''.join(frames))
sound_file.close()