"""
Aligned Dataset
"""

import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
from .pre_process import build_transforms


class AlignedDataset(Dataset):
	def __init__(self, opts, test_flag=False):
		super(AlignedDataset, self).__init__()
		self.test_flag = test_flag

		mode = opts['mode']

		img_paths_A = sorted(glob.glob(opts['pathA'] + '*.jpg'))
		img_paths_B = sorted(glob.glob(opts['pathB'] + '*.jpg'))

		lenA = len(img_paths_A)
		lenB = len(img_paths_B)
		len_ = min(lenA, lenB)

		img_paths_A = img_paths_A[:len_]
		img_paths_B = img_paths_B[:len_]

		self.img_paths_input = img_paths_A if mode == 'A2B' else img_pathsB
		self.img_paths_output = img_paths_B if mode == 'A2B' else img_pathsA

		self.transforms = build_transforms([
			'ToTensor'
		])

		self.len = len_
		return
	
	def __len__(self):
		return self.len

	def __getitem__(self, index):
		path_input = self.img_paths_input[index]
		path_output = self.img_paths_output[index]

		img_input = cv2.imread(path_input)
		img_output = cv2.imread(path_output)

		img_input = self.transforms(img_input)
		img_output = self.transforms(img_output)
		
		return img_input, img_output


if __name__ == '__main__':
	import glob
	import cv2
	import numpy as np

	paths = glob.glob('/data2/wangxinzhe/codes/datasets/CAVE/hsi/*.npy')
	print(paths)
	for i, x in enumerate(paths):
		gt = np.load(x)
		print('gt shape', gt.shape)
		print('gt min:', gt.min())
		print('gt max:', gt.max())

		cv2.imwrite('img/cave/%s.png'%(x.split('/')[-1].split('.')[0]), np.mean(gt, axis=2)/ gt.max() * 255 )
