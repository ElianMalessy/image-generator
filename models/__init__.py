import torch

torch.set_float32_matmul_precision('high')
dim_size = 64
size = dim_size * dim_size * 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
