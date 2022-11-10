"""
Core for training assembly
"""
import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from models.cycle_gan import CycleGANWarpper

class CycleGANCore:
	def __init__(self, batch_log=10):
		self.parallel = False
		self.batch_log = batch_log
		self.batch_cnt = 0
		self.epoch_cnt = 0
		self.scheduler = None
		pass

	def inject_logger(self, logger):
		self.logger = logger

	def inject_writer(self, writer):
		self.writer = writer

	def inject_device(self, device):
		self.device=device

	def build_model(self, opts):
		self.model = CycleGANWarpper(
			opts=opts,
			device=self.device
		)
		return
	
	def parallel(self, device_ids=['cuda:0']):
		if self.parallel==False:
			self.model = nn.DataParallel(self.model, device_ids=device_ids)
			self.optimizer = nn.DataParallel(self.optimizer, device_ids=args.device_ids)
			self.scheduler = nn.DataParallel(self.scheduler, device_ids=args.device_ids)

	def train(self, dataloader):
		if self.parallel==True:
			mean_loss = self.train_parallel(dataloader)
		else:
			mean_loss = self.train_single(dataloader)
		return mean_loss

	def train_single(self, dataloader):
		logger = self.logger
		writer = self.writer
		device = self.device
		model = self.model
		# scheduler = self.scheduler

		self.epoch_cnt += 1

		total_loss = {}
		G_lr = model.optimizer_G.state_dict()['param_groups'][0]['lr']
		D_lr = model.optimizer_D.state_dict()['param_groups'][0]['lr']
		logger.info('  lr_G:%f lr_D:%f'%(G_lr, D_lr))
		writer.add_scalar(tag='train/lr_G', scalar_value=G_lr, global_step=self.epoch_cnt)
		writer.add_scalar(tag='train/lr_D', scalar_value=D_lr, global_step=self.epoch_cnt)

		for i, (img_A, img_B) in enumerate(tqdm(dataloader)):
			img_A = img_A.to(device)
			img_B = img_B.to(device)

			loss = model.optimize(img_A, img_B)

			for k, v in loss.items():
				if not k in total_loss.keys():
					total_loss[k] = []
				total_loss[k].append(v)

			self.batch_cnt += 1
			if i % self.batch_log == 1:
				logger.info('  batch:%d ' % (i) +  str(loss))
				for k, v in loss.items():
					writer.add_scalar(tag='train/'+k, scalar_value=v, global_step=self.batch_cnt)
			pass
		mean_loss = {}
		for k, v in total_loss.items():
			mean_loss[k] = np.mean(v)
		logger.info('[Train] epoch:%d mean_loss ' % (i)+str(mean_loss))
		# scheduler.step(mean_loss)
		return 

	def train_parallel(self, dataloader):
		logger = self.logger
		device = self.device
		model = self.model
		loss_fn = self.loss_fn
		optimizer = self.optimizer
		scheduler = self.scheduler

		total_loss = []
		c_lr = optimizer.module.state_dict()['param_groups'][0]['lr']
		logger.info('  lr:%f'%(c_lr))

		for i, (lr, hr) in enumerate(tqdm(dataloader)):
			lr = lr.to(device)
			hr = hr.to(device)

			optimizer.zero_grad()
			pred = model(lr)
			loss = loss_fn(pred, hr)
			loss.backward()
			optimizer.module.step()

			total_loss.append(loss.item())
			if i % self.batch_log == 1:
				logger.info('  batch:%d loss:%.5f' % (i, loss.item()))
			pass
		mean_loss = np.mean(total_loss)
		scheduler.module.step(mean_loss)
		return mean_loss

	def visual(self, dataloader, img_num, save_path):
		# create save dir
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		# infer and save
		device = self.device
		model = self.model
		it = iter(dataloader)
		for i in range(min(img_num, dataloader.__len__())):
			img_A, img_B = next(it)
			assert len(img_A) == 1 and len(img_B) == 1, Exception('Test batch_size should be 1, not:%d' %(len(img_A)))
			img_A = img_A.to(device)
			img_B = img_B.to(device)
			with torch.no_grad():
				A2B, B2A = model.forward(img_A, img_B)
			# torch->numpy; 1CHW->HWC; [0, 1]->[0, 255]
			# img_A = img_A.cpu().numpy()[0].transpose(1, 2, 0) * 255
			# img_B = img_B.cpu().numpy()[0].transpose(1, 2, 0) * 255
			A2B_ = A2B.cpu().numpy()[0].transpose(1, 2, 0) * 255
			B2A_ = B2A.cpu().numpy()[0].transpose(1, 2, 0) * 255
			# save images
			# cv2.imwrite(save_path+'%d_img_A', img_A)
			# cv2.imwrite(save_path+'%d_img_A', img_A)
			cv2.imwrite(save_path+'%d_A2B.png'%(i), A2B_)
			cv2.imwrite(save_path+'%d_B2A.png'%(i), B2A_)
		return


	def save_ckpt(self, save_path):
		torch.save({
			'model': self.model
		}, save_path)
		return
	
	def load_ckpt(self, path):
		ckpt = torch.load(path)
		self.model = ckpt['model'].to(self.device)
		return

	def predict(self, dataloader):
		device = self.device
		model = self.model

		err_channel = None

		for i, (lr, hr) in enumerate(tqdm(dataloader)):
			lr = lr.to(device)
			hr = hr.to(device)

			with torch.no_grad():
				pred = model(lr)
			# cal error
			pred_ = pred.cpu().numpy()
			hr_ = hr.cpu().numpy()
			err_ = np.abs(pred_ - hr_)
			err_c = np.mean(err_, axis=(2, 3))
			if err_channel is None:
				err_channel = err_c
			else:
				err_channel = np.concatenate((err_channel, err_c), axis=0)
		return err_channel
