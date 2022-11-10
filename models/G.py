"""
Generator
"""
import torch
import torch.nn as nn
from torch.nn import init


def build_G():
	model = ResNet(channels=3, n_feats=64)
	return model


class ResNet(nn.Module):
		"""Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
		We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
		"""

		def __init__(self, channels, n_feats=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
				"""Construct a Resnet-based generator
				Parameters:
						input_nc (int)      -- the number of channels in input images
						output_nc (int)     -- the number of channels in output images
						ngf (int)           -- the number of filters in the last conv layer
						norm_layer          -- normalization layer
						use_dropout (bool)  -- if use dropout layers
						n_blocks (int)      -- the number of ResNet blocks
						padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
				"""
				assert(n_blocks >= 0)
				super(ResNet, self).__init__()

				use_bias = True
				n_downsampling = 2
				
				# head
				model = [
					nn.ReflectionPad2d(3),
					nn.Conv2d(channels, n_feats, kernel_size=7, padding=0, bias=use_bias),
					norm_layer(n_feats),
					nn.ReLU(inplace=True)
				]

				# down sample
				for i in range(n_downsampling):  # add downsampling layers
						mult = 2 ** i
						model += [
							nn.Conv2d(n_feats * mult, n_feats * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
							norm_layer(n_feats * mult * 2),
							nn.ReLU(inplace=True)]
				mult = 2 ** n_downsampling

				# res block
				for i in range(n_blocks):       # add ResNet blocks
						model += [
							ResnetBlock(n_feats * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

				# up sample
				for i in range(n_downsampling):  # add upsampling layers
						mult = 2 ** (n_downsampling - i)
						model += [
							nn.ConvTranspose2d(n_feats * mult, int(n_feats * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
							norm_layer(int(n_feats * mult / 2)),
							nn.ReLU(inplace=True)]

				# tail
				model += [
					nn.ReflectionPad2d(3),
					nn.Conv2d(n_feats, channels, kernel_size=7, padding=0),
					nn.Tanh()
				]

				# build model
				self.model = nn.Sequential(*model)

		def forward(self, input):
				"""Standard forward"""
				return self.model(input)


class ResnetBlock(nn.Module):
		"""Define a Resnet block"""

		def __init__(self, channels, padding_type, norm_layer, use_dropout, use_bias):
				"""Initialize the Resnet block
				A resnet block is a conv block with skip connections
				We construct a conv block with build_conv_block function,
				and implement skip connections in <forward> function.
				Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
				"""
				super(ResnetBlock, self).__init__()
				self.conv_block = self.build_conv_block(channels, padding_type, norm_layer, use_dropout, use_bias)

		def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
				"""Construct a convolutional block.
				Parameters:
						dim (int)           -- the number of channels in the conv layer.
						padding_type (str)  -- the name of padding layer: reflect | replicate | zero
						norm_layer          -- normalization layer
						use_dropout (bool)  -- if use dropout layers.
						use_bias (bool)     -- if the conv layer uses bias or not
				Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
				"""
				conv_block = []
				p = 0
				if padding_type == 'reflect':
						conv_block += [nn.ReflectionPad2d(1)]
				elif padding_type == 'replicate':
						conv_block += [nn.ReplicationPad2d(1)]
				elif padding_type == 'zero':
						p = 1
				else:
						raise NotImplementedError('padding [%s] is not implemented' % padding_type)

				conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
				if use_dropout:
						conv_block += [nn.Dropout(0.5)]

				p = 0
				if padding_type == 'reflect':
						conv_block += [nn.ReflectionPad2d(1)]
				elif padding_type == 'replicate':
						conv_block += [nn.ReplicationPad2d(1)]
				elif padding_type == 'zero':
						p = 1
				else:
						raise NotImplementedError('padding [%s] is not implemented' % padding_type)
				conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

				return nn.Sequential(*conv_block)

		def forward(self, x):
				"""Forward function (with skip connections)"""
				out = x + self.conv_block(x)  # add skip connections
				return out
