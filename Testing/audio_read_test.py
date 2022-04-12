import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave

# Plays a given wav file
def playback(file):
    # Initialize audio object and open wav file
    audio = pyaudio.PyAudio()
    rec = wave.open(file, "rb")

    # Create the stream for audio output
    stream = audio.open(format=audio.get_format_from_width(rec.getsampwidth()),
                        channels=rec.getnchannels(),
                        rate=rec.getframerate(),
                        output=True)

    # Store the audio file contents into an array
    data = rec.readframes(1024)

    # Play the file
    print("Playing file...")
    while len(data) > 0:
        stream.write(data)
        data = rec.readframes(1024)

def waveform(file):
    # Open the wav file
    rec = wave.open(file, "rb")

    # Code for stereo wav files
    if rec.getnchannels() == 2:
        # Read the audio data into an array
        signal = rec.readframes(-1)

        # Convert the signal array into raw sample values
        signal = np.frombuffer(signal, dtype="int16")

        # Split the signal into left and right channels
        left_channel = signal[0::2]
        right_channel = signal[1::2]

        # Get the frame rate and create the corresponding time array
        f_rate = rec.getframerate()
        left_time = np.linspace(0,
                        len(left_channel) / f_rate,
                        num = len(left_channel))
        right_time = np.linspace(0,
                        len(right_channel) / f_rate,
                        num = len(right_channel))

        # Plot the waveforms
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
    # Code for mono wav files
    else:
        # Read the audio data into an array
        signal = rec.readframes(-1)

        # Convert the signal array into raw sample values
        signal = np.frombuffer(signal, dtype="int16")

        # Get the frame rate and create the corresponding time array
        f_rate = rec.getframerate()
        time = np.linspace(0,
                        len(signal) / f_rate,
                        num = len(signal))

        # Plot the waveform
        plt.figure(1)
        plt.title("Sound Wave")
        plt.ylabel("Sample Value")
        plt.xlabel("Time(s)")
        plt.plot(time, signal)
        plt.show()


######################################################################################

if __name__ == '__main__':
    file = "testRecord.wav"
    # playback(file)
    waveform(file)
