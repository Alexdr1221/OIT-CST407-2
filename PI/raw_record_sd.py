from asyncore import write
import numpy as np
import sounddevice as sd
import soundfile as sf

def rec_audio(duration, device, channels, sample_rate):
    #    print("Recording audio")
    data = sd.rec(int(duration*sample_rate), samplerate=sample_rate,
                  channels=channels, device=device)

    sd.wait()

#    print("Done recording")
    
    return data;

def write_file(filename, data, samplerate):
    print("Writing sound file: " + filename)
    sf.write(filename, data, samplerate, 'PCM_16')
    print("Done")


if __name__ == "__main__":
    duration = 1
    device = 0
    channels = 2
    sample_rate = 44100


    while 1:
        data = rec_audio(duration, device, channels, sample_rate)
        # write_file("sound.wav", data, sample_rate)
        print(np.sqrt(np.mean(data*data)))
