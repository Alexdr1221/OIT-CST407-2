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

    if rec.getnchannels() == 2:
        signal = rec.readframes(-1)
        signal = np.frombuffer(signal, dtype="int16")
        left_channel = signal[0::2]
        right_channel = signal[1::2]

        f_rate = rec.getframerate()

        left_time = np.linspace(0,
                        len(left_channel) / f_rate,
                        num = len(left_channel))
        right_time = np.linspace(0,
                        len(right_channel) / f_rate,
                        num = len(right_channel))

        plt.figure(1)
        plt.title("Left Channel")
        plt.ylabel("Amplitude(dB)")
        plt.xlabel("Time(s)")
        plt.plot(left_time, left_channel)
        plt.grid(b=True)

        plt.figure(2)
        plt.title("Right Channel")
        plt.ylabel("Amplitude(dB)")
        plt.xlabel("Time(s)")
        plt.plot(right_time, right_channel)
        plt.grid(b=True)
        plt.show()

    else:
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

file = "timeShifted.wav"
# playback(file)
waveform(file)
