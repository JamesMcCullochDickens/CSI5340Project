import torch
import torch.nn as nn
import torchvision

class SimCLR(nn.Module):
	"""SimCLR model with contrastive learning."""

	def __init__(self, encoder, projection_dim, n_features):
		super(SimCLR, self).__init__()

		# Encoer to obtain h_i
		self.encoder = encoder
		self.n_features = n_features

		# MLP with one hidden layer to obtain z_i
		self.projector = nn.Sequential(
			nn.Linear(self.n_features, self.n_features, bias = False),
			nn.ReLU(),
			nn.Linear(self.n_features, projection_dim, bias = False))

		self.encoder.fc = nn.Identity()


	def forward(self, x_i, x_j):
		h_i = self.encoder(x_i)
		h_j = self.encoder(x_j)

		z_i = self.projector(h_i)
		z_j = self.projector(h_j)

		return h_i, h_j, z_i, z_j



