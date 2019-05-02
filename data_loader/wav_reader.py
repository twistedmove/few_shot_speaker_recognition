import librosa
import os
import numpy as np
from scipy.signal import lfilter, butter

# import sigproc
import data_loader.sigproc as sigproc


def load_wav(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()
	return audio


def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1,-1], [1,-alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


def get_fft_spectrum(filename, config, mode):
	signal = load_wav(filename, config['signal_processing']['SAMPLE_RATE'])
	signal *= 2**15

	# get FFT spectrum
	signal = remove_dc_and_dither(signal, config['signal_processing']['SAMPLE_RATE'])
	signal = sigproc.preemphasis(signal, coeff=config['signal_processing']['PREEMPHASIS_ALPHA'])
	frames = sigproc.framesig(signal, frame_len=config['signal_processing']['FRAME_LEN']*config['signal_processing']['SAMPLE_RATE'], 
								frame_step=config['signal_processing']['FRAME_STEP']*config['signal_processing']['SAMPLE_RATE'], winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=config['signal_processing']['NUM_FFT']))
	fft_norm = normalize_frames(fft.T)

	rsize = 300
	if mode=='train':
		rstart = np.random.randint(0, high=(fft_norm.shape[1]-rsize))
	else:
		rstart = int((fft_norm.shape[1]-rsize)/2)
	out = fft_norm[:,rstart:rstart+rsize]
	return out
