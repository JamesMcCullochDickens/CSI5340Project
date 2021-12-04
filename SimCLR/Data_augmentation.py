import torchvision

class TransformSimCLR:
	"""Transform any given sample to two correlated views, return x_i, x_j."""
	def __init__(self, size, s = 1):
		color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

		# data augmentation for SimCLR input
		self.data_transforms = torchvision.transforms.Compose([
			torchvision.transforms.RandomResizedCrop(size=size),
			torchvision.transforms.RandomHorizontalFlip(), # 0.5 prob
			torchvision.transforms.RandomApply([color_jitter], p=0.8),
			torchvision.transforms.RandomGrayscale(p=0.2),
			torchvision.transforms.ToTensor()])

		# no augmentation except for resize
		self.data_transforms_test = torchvision.transforms.Compose([
			torchvision.transforms.Resize(size=size),
			torchvision.transforms.ToTensor()])

	def __call__(self, x):
		return self.data_transforms(x), self.data_transforms(x)