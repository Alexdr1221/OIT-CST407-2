import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave

def playback(file):
    audio = pyaudio.PyAudio()
    rec = wave.open(file, "rb")

    stream = audio.open(format=audio.get_format_from_width(rec.getsampwidth()),
                        channels=rec.getnchannels(),
                        rate=rec.getframerate(),
                        output=True)

    data = rec.readframes(1024)

    print("Playing file...")
    while len(data) > 0:
        stream.write(data)
        data = rec.readframes(1024)

def waveform(file):
    rec = wave.open(file, "rb")

    signal = rec.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    f_rate = rec.getframerate()

    time = np.linspace(0,
                       len(signal) / f_rate,
                       num = len(signal))

    plt.figure(1)
    plt.title("Sound Wave")
    plt.ylabel("Amplitude(dB)")
    plt.xlabel("Time(s)")
    plt.plot(time, signal)
    plt.show()


######################################################################################

file = "testRecord.wav"

waveform(file)
# playback(file)
