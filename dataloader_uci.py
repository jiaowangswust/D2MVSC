import torch
from dataset_uci import UCI
data_loader_train = torch.utils.data.DataLoader(UCI(), batch_size=2000, shuffle=False)


