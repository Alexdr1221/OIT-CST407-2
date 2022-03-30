import pyaudio
import wave

# Settings
channels = 1
sample_rate = 44100
frame_size = 1024

# Init Audio Object+
audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16,         # Samples sotred in 16 bit number
                    channels=channels,              # Stereo vs Mono
                    rate=sample_rate,               # Sampling Rate, 44khz
                    input=True,                     # Input stream
                    frames_per_buffer=frame_size)

# Recorded audio stored as frames
frames = []

try:
    print("Recording...")
    while True:
        data = stream.read(frame_size)
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
sound_file.setnchannels(channels)
sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
sound_file.setframerate(sample_rate)
sound_file.writeframes(b''.join(frames))
sound_file.close()