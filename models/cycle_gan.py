"""
Wapper for CycleGAN model
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .loss import GANLoss
from .D import build_D
from .G import build_G

class CycleGANWarpper():
	def __init__(self, opts, device):
		self.device = device

		mode = opts['mode']
		gan_mode = opts['gan_mode']
		beta1 = opts['beta1']
		lr = opts['lr']
		self.mode = mode
		self.p_A = opts['p_A']
		self.p_B = opts['p_B']
		self.p_Idt = opts['p_Idt']

		# G
		self.G_A = build_G().to(device)
		self.G_B = build_G().to(device)

		if mode == 'train':
			# D
			self.D_A = build_D().to(device)
			self.D_B = build_D().to(device)
			# loss
			self.loss_GAN = GANLoss(mode=gan_mode).to(device)
			self.loss_Cycle = nn.L1Loss()
			self.loss_Idt = nn.L1Loss()
			# optim
			self.optimizer_G = optim.Adam([
				{'params': self.G_A.parameters()},
				{'params': self.G_B.parameters()}
			], lr=lr, betas=(beta1, 0.999))

			self.optimizer_D = optim.Adam([
				{'params': self.D_A.parameters()},
				{'params': self.D_B.parameters()}
			], lr=lr, betas=(beta1, 0.999))
		return

	def forward(self, input_img, output_img):
		self.real_A = input_img
		self.fake_B = self.G_A(input_img)
		self.rec_A = self.G_B(self.fake_B)
		self.real_B = output_img
		self.fake_A = self.G_B(output_img)
		self.rec_B = self.G_A(self.fake_A)
		return self.fake_B, self.fake_A

	def backward_D_basic(self, D, real, fake):
		# real
		pred_real = D(real)
		loss_D_real = self.loss_GAN(pred_real, True)
		# fake
		pred_fake = D(fake.detach())
		loss_D_fake = self.loss_GAN(pred_fake, False)
		# loss, backward
		loss = (loss_D_real + loss_D_fake) * 0.5
		loss.backward()
		return loss

	def backward_D_A(self):
		self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, self.fake_B)
		return
	
	def backward_D_B(self):
		self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, self.fake_A)
		return
	
	def backward_G(self):
		# identity loss
		if self.p_Idt > 0:
			self.idt_A = self.G_A(self.real_B)
			self.loss_idt_A = self.loss_Idt(self.idt_A, self.real_B) * self.p_B * self.p_Idt
			self.idt_B = self.G_B(self.real_A)
			self.loss_idt_B = self.loss_Idt(self.idt_B, self.real_A) * self.p_A * self.p_Idt
		else:
			self.loss_idt_A = 0
			self.loss_idt_B = 0
		# loss for G
		self.loss_G_A = self.loss_GAN(self.D_A(self.fake_B), True)
		self.loss_G_B = self.loss_GAN(self.D_B(self.fake_A), True)
		# cycle loss
		self.loss_cycle_A = self.loss_Cycle(self.rec_A, self.real_A) * self.p_A
		self.loss_cycle_B = self.loss_Cycle(self.rec_B, self.real_B) * self.p_B
		# combined loss
		self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
		self.loss_G.backward()
		return
	
	def optimize(self, img_A, img_B):
		# forward
		self.forward(img_A, img_B)
		# G_A, G_B
		self.set_requires_grad([self.D_A, self.D_B], False)
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()
		# D_A, D_B
		self.set_requires_grad([self.D_A, self.D_B], True)
		self.optimizer_D.zero_grad()
		self.backward_D_A()
		self.backward_D_B()
		self.optimizer_D.step()
		# out
		out = {
			'loss_G_A':self.loss_G_A.item(),
			'loss_G_B':self.loss_G_B.item(),
			'loss_cycle_A': self.loss_cycle_A.item(),
			'loss_cycle_B': self.loss_cycle_B.item(),
			'loss_idt_A': 0 if self.loss_idt_A == 0 else self.loss_idt_A.item(),
			'loss_idt_B': 0 if self.loss_idt_B == 0 else self.loss_idt_B.item(),
			'loss_G':self.loss_G.item(),
			'loss_D_A':self.loss_D_A.item(),
			'loss_D_B':self.loss_D_B.item(),
		}
		return out

	def set_requires_grad(self, nets, requires_grad=False):
		nets = nets if isinstance(nets, list) else [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad
		return