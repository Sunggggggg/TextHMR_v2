import torch
import torch.nn as nn

class L2Norm(nn.Module):
	def forward(self, x):
		return x / x.norm(dim=-1, keepdim=True)
	
class ConCatModule(nn.Module):
	def __init__(self):
		super(ConCatModule, self).__init__()

	def forward(self, x):
		x = torch.cat(x, dim=1)
		return x
	
class TIRG(nn.Module):
	"""
	The TIRG model.
	Implementation derived (except for BaseModel-inherence) from
	https://github.com/google/tirg (downloaded on July 23th 2020).
	The method is described in Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia
	Li, Li Fei-Fei, James Hays. "Composing Text and Image for Image Retrieval -
	An Empirical Odyssey" CVPR 2019. arXiv:1812.07119
	"""

	def __init__(self, input_dim=[512, 512], output_dim=512, out_l2_normalize=False):
		super(TIRG, self).__init__()

		self.input_dim = sum(input_dim)
		self.output_dim = output_dim

		# --- modules
		self.a = nn.Parameter(torch.tensor([1.0, 1.0])) # changed the second coeff from 10.0 to 1.0
		self.gated_feature_composer = nn.Sequential(
				ConCatModule(), nn.BatchNorm1d(self.input_dim), nn.ReLU(),
				nn.Linear(self.input_dim, self.output_dim))
		self.res_info_composer = nn.Sequential(
				ConCatModule(), nn.BatchNorm1d(self.input_dim), nn.ReLU(),
				nn.Linear(self.input_dim, self.input_dim), nn.ReLU(),
				nn.Linear(self.input_dim, self.output_dim))

		if out_l2_normalize:
			self.output_layer = L2Norm() # added to the official TIRG code
		else:
			self.output_layer = nn.Sequential()

	def forward(self, main_features, modifying_features):
		"""
		main_features		: [B, T, dim]
		modifying_features	: [B, T, dim]
		"""
		f1 = self.gated_feature_composer((main_features, modifying_features))
		f2 = self.res_info_composer((main_features, modifying_features))
		f = torch.sigmoid(f1) * main_features * self.a[0] + f2 * self.a[1]
		f = self.output_layer(f)
		return f