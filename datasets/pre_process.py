"""
Pre Process
"""
import torchvision.transforms as transforms

def build_transforms(opts):
	trans = []
	for opt in opts:
		if opt == 'ToTensor':
			trans.append(transforms.ToTensor())	# HWC->CHW; [0, 255]->[0, 1]; np.ndarray->torch.FloatTensor
	return transforms.Compose(trans)