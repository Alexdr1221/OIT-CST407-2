import pyaudio
import wave

# Init Audio Object
audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16, # Samples sotred in 16 bit number
                    channels=1,             # Stereo vs Mono
                    rate=44100,             # Sampling Rate, 44khz
                    input=True,             # Input stream
                    frames_per_buffer=1024) # ???

# Recorded audio stored as frames
frames = []

try:
    print("Recording...")
    while True:
        data = stream.read(1024)
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
sound_file.setnchannels(1)
sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
sound_file.setframerate(44100)
sound_file.writeframes(b''.join(frames))
sound_file.close()