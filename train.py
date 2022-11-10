import os
import cv2
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataset
from utils.logger import create_logger
from utils.seed import set_seed
from utils.core import CycleGANCore

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=4, type=int)
	parser.add_argument('--epoch', default=200)
	parser.add_argument('--lr', default=2e-4, type=float, help='Learning Rate')
	parser.add_argument('--seed', default=17, type=int)
	parser.add_argument('--device', default='cuda:7')
	parser.add_argument('--parallel', default=False)
	parser.add_argument('--device_ids', default=['cuda:5', 'cuda:6', 'cuda:7'])
	# Dataset
	parser.add_argument('--mode', default='Season')
	parser.add_argument('--train_path_A', default='/data2/wangxinzhe/codes/datasets/summer2winter_yosemite/trainA/')
	parser.add_argument('--train_path_B', default='/data2/wangxinzhe/codes/datasets/summer2winter_yosemite/trainB/')
	parser.add_argument('--test_path_A', default='/data2/wangxinzhe/codes/datasets/summer2winter_yosemite/testA/')
	parser.add_argument('--test_path_B', default='/data2/wangxinzhe/codes/datasets/summer2winter_yosemite/testB/')
	args = parser.parse_args()
	print(args)
	return args

def train(args):
	t = time.strftime('%Y-%m-%d_%H:%M:%S')
	checkpoint_path = 'checkpoints/%s/%s/' % (args.mode,  t)
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	log_path = 'log/%s/' %(args.mode)
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	logger = create_logger(log_path + '%s.log'%(t))
	logger.info(str(args))

	writer = SummaryWriter('tensorboard/%s/%s/' % (args.mode,  t))

	# set seed
	set_seed(args.seed)

	# device
	cudnn.benchmark = True
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	# dataset
	dataset, dataloader = build_dataset(
		opts={
			'mode':'A2B', 
			'pathA':args.train_path_A, 
			'pathB':args.train_path_B 
		}, batch_size=args.batch_size, test_flag=False)
	test_dataset, test_dataloader = build_dataset(
		opts={
			'mode':'A2B', 
			'pathA':args.test_path_A, 
			'pathB':args.test_path_B
		}, batch_size=1, test_flag=True)

	# core
	core = CycleGANCore(batch_log=10)
	core.inject_logger(logger)
	core.inject_writer(writer)
	core.inject_device(device)
	opts = {
		'mode':'train',
		'gan_mode':'L2',	# ['L2', 'BCE', 'wgangp']
		'p_A': 0.5,
		'p_B': 0.5,
		'p_Idt': 0.5,
		'lr':args.lr,
		'beta1': 0.9,
	}
	logger.info(str(opts))
	core.build_model(opts=opts)

	if args.parallel:
		core.parallel(device_ids = args.device_ids)

	# train loop
	for epoch in range(args.epoch):
		core.train(dataloader)
		# core.test(test_dataloader)

		save_path = checkpoint_path + 'epoch=%d'%(epoch)
		core.visual(test_dataloader, img_num=10, save_path=save_path + '/')
		core.save_ckpt(save_path +'ckpt.pt')
	return


if __name__=='__main__':
	args = parse_args()
	train(args)