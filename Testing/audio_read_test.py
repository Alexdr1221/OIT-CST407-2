import pyaudio
import wave

audio = pyaudio.PyAudio()
rec = wave.open("testRecord.wav", "rb")

stream = audio.open(format=audio.get_format_from_width(rec.getsampwidth()),
                    channels=rec.getnchannels(),
                    rate=rec.getframerate(),
                    output=True)

data = rec.readframes(1024)

print("Playing file...")
while len(data) > 0:
    stream.write(data)
    data = rec.readframes(1024)