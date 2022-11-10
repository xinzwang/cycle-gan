"""
Build Dataset and dataloader
"""
from .aligned import AlignedDataset

from torch.utils.data.dataloader import DataLoader


# def SelectDatasetObject(name):
# 	if name in ['Pavia', 'PaviaU', 'Salinas', 'KSC', 'Indian', 'CAVE']:
# 		return SingleDataset
# 	elif name in ['ICVL']:
# 		return MultiDataset
# 	else:
# 		raise Exception('Unknown dataset:', name)

def build_dataset(opts, batch_size=32, test_flag=False):
	dataset = AlignedDataset(
		opts,
		test_flag=test_flag,
	)
	dataloader = DataLoader(
		dataset=dataset,
		batch_size=batch_size if not test_flag else 1,
		num_workers=8,
		shuffle= (not test_flag)	# shuffle only train
	)
	return dataset, dataloader
