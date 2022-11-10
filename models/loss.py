import torch
import torch.nn as nn


class GANLoss(nn.Module):
	def __init__(self, mode):
		super(GANLoss, self).__init__()
		self.mode = mode

		self.register_buffer('real_label', torch.tensor(1.0))
		self.register_buffer('fake_label', torch.tensor(0.0))

		if mode == 'L2':
			self.loss = nn.MSELoss()
		elif mode == 'BCE':
			self.loss = nn.BCEWithLogistsLoss()
		elif mode == 'wgangp':
			self.loss = None
		else:
			raise NotImplementedError('[GANLoss] mode %s not implemented' % mode)

	def get_target_tensor(self, pred, target_is_real):
		pass

	def __call__(self, pred, is_real):
		if self.mode in ['L2', 'BCE']:
			target = (self.real_label if is_real else self.fake_label).expand_as(pred)
			loss = self.loss(pred, target)
		elif self.mode == 'wgangp':
			loss = pred.mean() * (-1 if is_real else 1)
		return loss


