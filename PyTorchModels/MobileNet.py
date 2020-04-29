import torch
import torch.nn as nn

class Mobilev1Block(nn.Module):
	def __init__(self, in_channel, out_channel, first = False):
		super(Mobilev1Block, self).__init__()
		self.downSample = not (in_channel == out_channel) and not first
		self.dwConv1  = self.Fetch_DepthwiseConv(in_channel = in_channel,
												 out_channel = in_channel,
												 kernel_size = 3,
												 padding = 1,
												 stride = self.downSample + 1)
		self.stdConv2 = self.Fetch_StandardConv(in_channel = in_channel,
												out_channel = out_channel,
												kernel_size = 1,
												padding = 0,
												stride = 1)
	
	def forward(self, x):
		x = self.dwConv1(x)
		x = self.stdConv2(x)
		return x

	def Fetch_StandardConv(self, in_channel, out_channel, 
						   kernel_size = 3,
						   padding = 1,
						   stride = 1):
		layer  = [nn.Conv2d(in_channels = in_channel,
						    out_channels = out_channel,
							kernel_size = kernel_size,
							stride = stride,
							padding = padding),
				  nn.BatchNorm2d(out_channel),
				  nn.ReLU(inplace = True)]
		return nn.Sequential(* layer)

	def Fetch_DepthwiseConv(self, in_channel, out_channel,
						  	kernel_size = 3, 
						  	padding = 1, 
						  	stride = 1):
		layer  = [nn.Conv2d(in_channels = in_channel,
							out_channels = out_channel,
							kernel_size = kernel_size,
					 		stride = stride, 
							padding = padding,
							groups = in_channel),
				  nn.BatchNorm2d(out_channel),
				  nn.ReLU(inplace = True)]
		return nn.Sequential(* layer)


class MobileNetv1(nn.Module):
	model_name = 'MobileNetv1'
	def __init__(self, widthMultiplier = 1., num_of_classes = 10):
		'''
			MobileNetv1 has 2 hyperparemeters to shrink the model:
				Width Multiplier:		Shrink spectral size uniformly across all layers
				Resolution Multiplier:	Shrink spatial size uniformly across all layers (Not Supported)
		'''
		super(MobileNetv1, self).__init__()
		self._basic_depth = int(32 * widthMultiplier)
		self.conv1   = nn.Sequential(nn.Conv2d(3, self._basic_depth, kernel_size = 3, stride = 2, padding = 1),
				  					 nn.BatchNorm2d(self._basic_depth),
				  					 nn.ReLU(inplace = True))
		self.block1  = Mobilev1Block(self._basic_depth, 2 * self._basic_depth, True)
		self.block2  = Mobilev1Block(2 * self._basic_depth,  4 * self._basic_depth)
		self.block3  = Mobilev1Block(4 * self._basic_depth,  4 * self._basic_depth)
		self.block4  = Mobilev1Block(4 * self._basic_depth,  8 * self._basic_depth)
		self.block5  = Mobilev1Block(8 * self._basic_depth,  8 * self._basic_depth)
		self.block6  = Mobilev1Block(8 * self._basic_depth, 16 * self._basic_depth)
		self.iden    = self._identity_blocks(16 * self._basic_depth, 5)
		self.block7  = Mobilev1Block(16 * self._basic_depth, 32 * self._basic_depth)
		self.block8  = Mobilev1Block(32 * self._basic_depth, 32 * self._basic_depth)
		self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc      = nn.Linear(32 * self._basic_depth, num_of_classes)

	def _identity_blocks(self, channel, num_block = 5):
		layers = []
		for _ in range(num_block):
			layers.append(Mobilev1Block(channel, channel))
		return nn.Sequential(* layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.block2(self.block1(x))
		x = self.block4(self.block3(x))
		x = self.block6(self.block5(x))
		x = self.iden(x)
		x = self.block8(self.block7(x))
		x = self.avgPool(x)
		x = x.view(-1, 1024)
		x = self.fc(x)
		return x