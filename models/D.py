"""
Distribution
"""

import torch
import torch.nn as nn
from .common import init_net

def build_D():
	model = Convs(channels=3, n_feats=64, kernel_size=4, n_layers=3)
	return init_net(model, init_gain=0.02, init_type='normal')

class Convs(nn.Module):
	def __init__(self, channels=3, n_feats=64, kernel_size=4, n_layers=3):
		super(Convs, self).__init__()


		model = [
			nn.Conv2d(channels, n_feats, kernel_size=kernel_size, stride=2, padding=1),
			nn.LeakyReLU(0.2, inplace=True)
		]

		for i in range(1, n_layers):
			model += [
				nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=2, padding=1, bias=False),
				nn.BatchNorm2d(n_feats),
				nn.LeakyReLU(0.2, inplace=True)
			]
		
		model +=  [
			nn.Conv2d(n_feats, 1, kernel_size=kernel_size, stride=1, padding=1)
		]

		self.model=nn.Sequential(*model)
		return
	
	def forward(self, x):
		return self.model(x)





