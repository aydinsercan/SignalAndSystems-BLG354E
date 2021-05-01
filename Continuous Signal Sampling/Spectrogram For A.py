from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from playsound import playsound
import wavio


fs, signal = wavfile.read('sound.wav')

signal1 = signal / np.max((signal))

window_size = 2000
step = 2000


spec_matrix = np.zeros(((signal1.shape[0]-window_size)//step,window_size//2))

for i in range((signal1.shape[0]-window_size)//step):
    fft1 = np.abs(np.fft.fft(signal1[i*step:i*step+window_size]))
    fft = fft1[0:window_size//2]
    spec_matrix[i,:] = fft

spec_matrix2 = 10*np.log(spec_matrix.transpose((1,0))+1e-5)

plt.pcolormesh(spec_matrix2)
ax = plt.gca()
ax.set_yscale('linear')
plt.show()

    
    
