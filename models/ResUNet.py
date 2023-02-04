import models.resnet_basicblock as B
import torch.nn as nn
import numpy as np


class ResUNet(nn.Module):
	def __init__(self, in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
		super(ResUNet, self).__init__()

		self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')
		# downsample
		downsample_block = B.downsample_strideconv	
		self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
		self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
		self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))
		self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

		# upsample 
		upsample_block = B.upsample_convtranspose
		self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

	def forward(self, x):
		h, w = x.size()[-2:]
		paddingBottom = int(np.ceil(h/8)*8-h)
		paddingRight = int(np.ceil(w/8)*8-w)
		x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

		x1 = self.m_head(x)
		x2 = self.m_down1(x1)
		x3 = self.m_down2(x2)
		x4 = self.m_down3(x3)
		x = self.m_body(x4)
		x = self.m_up3(x+x4)
		x = self.m_up2(x+x3)
		x = self.m_up1(x+x2)
		x = self.m_tail(x+x1)
		x = x[..., :h, :w]
		return x