import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, ifft, fftshift
import matplotlib
matplotlib.use("Agg")
import cv2
import moviepy.editor as mpe
from moviepy.editor import *
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt; plt.rcdefaults()

[data, samplerate]= sf.read('SilentKnight.wav')
shapex = data.shape
new_data = np.zeros(((math.ceil(shapex[0]/4410)*4410),2))
new_data[0:shapex[0],:] = data
tour_count = int(new_data.shape[0]/4410); 
#print(tour_count)
all_ffts = np.zeros((4410,(tour_count)))

i=1
while i<= tour_count:
	transform = np.fft.fftshift(np.fft.fft(new_data[(i-1)*4410:(i)*4410,0]))
	abs_transform = np.abs((transform.real)**2 + (transform.imag)**2)
	all_ffts[:,i-1] = 10*np.log10(abs_transform+1)
	i+=1
	
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,6))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 24.0, (1000,600))
index = np.arange(len(transform))

u=1
while u <= tour_count:	
	plt.bar(index,all_ffts[:,(u-1)])
	ax.plot(np.arange(1,101,1), np.random.rand(100,1))
	plt.savefig('snap.png', format='png')
	ax.clear()
	X = cv2.imread('snap.png')
	out.write(X)
	u+=1
	
out.release()
myclip = mpe.VideoFileClip('output.avi')
audiobackground = mpe.AudioFileClip('SilentKnight.wav')
finalclip = myclip.set_audio(audiobackground)
finalclip.write_videofile('total_output.mp4')

