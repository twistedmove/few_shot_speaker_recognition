from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import pandas as pd
from glob import glob
import data_loader.wav_reader as wav_reader
from torch.utils.data import Sampler

class SpeechDataLoader_support(BaseDataLoader):
	def __init__(self, partition, labels, batch_size, config):
		self.dataset = SpeechDataSet(partition, labels, config, 'test')
		super(SpeechDataLoader_support, self).__init__(self.dataset, None, batch_size,
												False,
												0.0,
												config['data_loader']['args']['num_workers']
												)
class SpeechDataLoader_query(BaseDataLoader):
	def __init__(self, partition, labels, config):
		self.dataset = SpeechDataSet(partition, labels, config, 'test')
		super(SpeechDataLoader_query, self).__init__(self.dataset, None, config['data_loader']['args']['batch_size'], False,0.0,
												 config['data_loader']['args']['num_workers'])

class SpeechDataSet(Dataset):
	def __init__(self, list_IDs, labels, config, mode):
		self.labels = labels
		self.list_IDs = list_IDs
		self.config = config
		self.mode = mode

	def __len__(self):
		return len(self.list_IDs)
	
	def __getitem__(self, index):
		ID = self.list_IDs[index]
		y = np.array(self.labels[ID]-1)
		y = torch.from_numpy(y).long()
		X = get_features(ID, self.config, self.mode)
		return X, y

def get_features(list_file, config, mode):
	features = wav_reader.get_fft_spectrum(list_file, config, mode)
	features = features.reshape(1,*features.shape)
	features = torch.from_numpy(features).float()
	return features
