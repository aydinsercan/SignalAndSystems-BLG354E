from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from playsound import playsound
import wavio

class Spectogram(object):
    def __init__(self):   
        import numpy as np
    
    def window(self, x, fs, window_dur_in_second, frame_shift):
        import numpy as np
        x = x[:,1]
        sound_length = len(x)
        window_length = int(fs*window_dur_in_second)
        shift_length = int(fs*frame_shift)
        num_window = int((sound_length-window_length)/shift_length + 1)
        windowed_data = []
        for i in range(int(num_window)):
            window = [0.54-0.46*np.cos((2*3.14*i)/(window_length-1)) for i in range(window_length)]
            frame = x[(i*shift_length):(i*shift_length)+window_length]*window
            windowed_data.append(frame)
        return windowed_data

    def STFT(self, data, N, fs, window_dur_in_second, frame_shift, plot):
        windowed_data = self.window(data,fs, window_dur_in_second , frame_shift = frame_shift )
        STFT = []
        dft_frame = []
        for i,frame in enumerate(windowed_data):            
            dft_frame = np.fft.fft(frame,N)
            STFT.append(dft_frame)
            if plot==True:
                f = np.arange(0,fs/2,fs/N)
                plt.plot(f,10*np.log10(np.abs(dft_frame[:int(len(dft_frame)/2)])))
                plt.grid(True)
                plt.title("FFT of Frame: {}".format(i))
                plt.xlabel("Freqency(Hz)")
                plt.figure()
        STFT = np.array(STFT)
        return STFT
      
    def plotSTFT(self, STFT, N ,fs):
        STFT = np.transpose(STFT)
        STFT = STFT[:int(N/2)][:]
        f = np.arange(0,fs/2,fs/N)
        f = f.reshape((f.shape[0],1))
        t = np.arange(0,STFT.shape[1],1)
        plt.pcolormesh(t, f, 10*np.log10(np.abs(STFT)))
        plt.title('Spectogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Frames')
        plt.yscale('symlog')
 #       plt.ylim(0, 100000)
        plt.show()
    
fs, data = wavfile.read('aphex_twin_equation.wav')
spec = Spectogram()
xSTFT = spec.STFT(data, 1024, fs, window_dur_in_second = 0.01, frame_shift = 0.005, plot = False)
spec.plotSTFT(xSTFT, 1024 ,fs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
