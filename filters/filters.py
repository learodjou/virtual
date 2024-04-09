"""This module contains functions to apply filters to signals."""

from scipy.fft import fft, fftfreq
from scipy.fftpack import fftshift
from scipy.signal import filtfilt, firwin, iirfilter, kaiserord, lfilter


def fourier_transform(signal, sample_rate=44100, duration=5):
    """Compute the Fourier Transform of a signal."""
    N = sample_rate * duration
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)

    # ordered fft
    yf = fftshift(yf)
    xf = fftshift(xf)
    return xf, yf


def iir_filter(signal, f_cutoff, f_sampling, fbf=False):
    """IR."""
    b, a = iirfilter(4, Wn=f_cutoff, fs=f_sampling, btype="low", ftype="butter")
    if not fbf:
        filtered = lfilter(b, a, signal)
    else:
        filtered = filtfilt(b, a, signal)

    return filtered


def fir_filter(signal, nyq_rate, cutoff_hz):
    """FIR."""
    width = 5.0 / nyq_rate
    ripple_db = 20.0
    N, beta = kaiserord(ripple_db, width)
    taps = firwin(N, cutoff_hz / nyq_rate, window=("kaiser", beta))
    filtered_x = lfilter(taps, 1.0, signal)
    return filtered_x, taps, N
