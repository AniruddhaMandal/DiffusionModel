import matplotlib.pyplot as plt 
import torch
from data_loader import Animal_10
from SmallDiffusion.DiffusionModel import Diffusion
from torch.utils.data import DataLoader

animal_data = Animal_10(transform=torch.tensor)
animal_dataloder = DataLoader(animal_data,32)
image,labels = next(iter(animal_dataloder))

model = Diffusion(32)
time = torch.zeros((32,1))
time = time+32
print(f"Time Input: {time.shape}")
print(f"Model Output: {model.forward(image,time).shape}")