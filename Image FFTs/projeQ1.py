import scipy.io.wavfile
from scipy.fftpack import dct
import cv2
import math
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import scipy
import pylab


image1 = cv2.imread ('lena_grayscale.jpg' , cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread ('fabric_grayscale.jpg' , cv2.IMREAD_GRAYSCALE)

#2DFft and fftshifting
im_fft1 = fftpack.fft2(image1)
fshift1 = np.fft.fftshift(im_fft1)
im_fft2 = fftpack.fft2(image2)
fshift2 = np.fft.fftshift(im_fft2)

#Magnitude and phase
magnitude_spectrum = 20*np.log(np.abs(fshift1))
phase_spectrum = np.angle(fshift1)

#Reconstruct
f_ishift1 = np.fft.ifftshift(fshift1)
img_back = np.fft.ifft2(f_ishift1)
img_back = np.abs(img_back)

"""
#plot
pylab.figure(figsize = (12,10))
pylab.subplot(2,2,1),pylab.imshow(image1, cmap = 'gray')
pylab.title('Original Image')
pylab.subplot(2,2,2),pylab.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude')
pylab.subplot(2,2,3),pylab.imshow(phase_spectrum, cmap = 'gray')
plt.title('Phase')
pylab.subplot(2,2,4),pylab.imshow(img_back, cmap = 'gray')
plt.title('Reconstructed Image')
pylab.show()
"""

combined = np.multiply(np.abs(im_fft2), np.exp(1j*np.angle(im_fft1)))
imgCombined = np.real(np.fft.ifft2(combined))
plt.imshow(imgCombined, cmap='gray')
plt.show()
cv2.imwrite('Lena_output2.png',imgCombined )
