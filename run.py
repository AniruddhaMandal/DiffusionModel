import matplotlib.pyplot as plt 
from data_loader import Animal_10
import torch

data = Animal_10()
for img,label in data:
    print(img.shape)
    print(label)
    break