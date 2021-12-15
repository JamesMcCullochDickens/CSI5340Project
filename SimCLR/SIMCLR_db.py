import torch
import torch.nn as nn
import torchvision

class SimCLR(nn.Module):
	"""
	SimCLR model with contrastive learning.
	Dual-encoder(rgb+depth/depth+depth).
	"""

	def __init__(self, encoder1, encoder2, projection_dim, n_features):
		super(SimCLR, self).__init__()

		# Encoer to obtain h_i
		self.encoder1 = encoder1
		self.encoder2 = encoder2
		self.n_features = n_features

		# MLP with one hidden layer to obtain z_i
		self.projector = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n_features, self.n_features, bias = False),
			nn.ReLU(),
			nn.Linear(self.n_features, projection_dim, bias = False))

		self.encoder1.fc = nn.Identity()
		self.encoder2.fc = nn.Identity()


	def forward(self, x_i, x_j):
		h_i = self.encoder1(x_i)
		h_j = self.encoder2(x_j)

		z_i = self.projector(h_i)
		z_j = self.projector(h_j)

		return h_i, h_j, z_i, z_j



