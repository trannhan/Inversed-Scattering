import numpy as np
import scipy.fftpack as fftp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def fourier_series(x, y, wn, n=None):
    # get FFT
    myfft = fftp.fft(y, n)
    # kill higher freqs above wavenumber wn
    myfft[wn:-wn] = 0
    # make new series
    y2 = fftp.ifft(myfft)

    plt.figure(num=None)
    plt.plot(x, y, x, y2)
    plt.show()

if __name__=='__main__':
    x = np.array([float(i) for i in range(0,360)])
    y = np.sin(2*np.pi/360*x) + np.sin(2*2*np.pi/360*x) + 5

    fourier_series(x, y, 3, 360)