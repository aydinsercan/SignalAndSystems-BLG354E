from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from playsound import playsound
import wavio
import math
import sounddevice as sd


def create_idft_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(2 * np.pi * 1J / N)
    W = np.power(omega, i * j)
    return 1/N*W


spectrogram = np.load('spectrogram.npy')
phase = np.load('phases.npy')


spectrogram_prepare = np.zeros((98,2000));

for i in range(len(spectrogram)):
    rev = spectrogram[i][::-1]
    add = rev[0:len(rev)-1]
    add = np.insert(add,0,add[0])
    res3 = np.concatenate((spectrogram[i],add))
    spectrogram_prepare[i] = [10**(x/10) for x in res3]
        

union_spec_phase = np.zeros((98,2000))
[r,c] = phase.shape

for row in range(r):
    for col in range(c):
        union_spec_phase[row][col] = spectrogram_prepare[row][col] * (np.e**(1j*phase[row][col])) 


idf_matrix = create_idft_matrix(2000)

result = []

for union in union_spec_phase:
    array = []
    for idf_row in idf_matrix:
        dot_product = np.multiply(union,idf_row)
        sumOf = sum(dot_product)
        array.append(sumOf)
    result.append(array)
    
        
voice_record = [item for sublist in result for item in sublist]
voice_record.pop(0)
voice_record = np.real(voice_record)


m = np.max(np.abs(voice_record))
sigf32 = (voice_record/m).astype(np.float32)
wavfile.write('sound.wav', 44100, sigf32)



    
    
    
    
    
    
    
    
    
    
    
    
