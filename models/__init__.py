import torch

dim_size = 64
size = dim_size * dim_size * 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
