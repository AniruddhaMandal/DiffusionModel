import pickle
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class CIFAR_100(Dataset):
    def __init__(self) -> None:
        self.file_path = "Data/cifar-100-python/train"
        with open(self.file_path,'rb') as f:
            unpickeled_data = pickle.load(f,encoding="bytes")
        
        self.images = np.array(unpickeled_data[b'data'])
        self.labels = np.array(unpickeled_data[b'fine_labels'])

    def __len__(self)->int:
        return len(self.labels)
    
    def __getitem__(self, index) -> tuple:
        image = self.images[index]
        image = image.reshape(3,32,32)
        label = self.labels[index]
        return image,label


class Animal_10(Dataset):
    def __init__(self,transform=None) -> None:
        self.file_path = "Data/Animals-10/raw-img"
        self.transform = transform
        self.labels = []
        self.img_file = []
        for dir in os.listdir(self.file_path):
            for img_file in os.listdir(os.path.join(self.file_path,dir)):
                self.labels.append(dir)
                self.img_file.append(os.path.join(self.file_path,dir,img_file))

    def __len__(self)->int:
        return len(self.labels)
    
    def __getitem__(self, index) -> tuple:
        label = self.labels[index]
        img_ = Image.open(self.img_file[index])
        img = np.array(img_)
        img_.close()
        if self.transform:
            img = self.transform(img)
        return img, label

    def __getitems__(self,indices):
        labels = self.labels[indices]
        imgs = []
        for img_f in self.img_file[indices]:
            img_ = Image.open(img_f)
            imgs.append(np.array(img_).transpose(2,0,1))
            img_.close()
            if self.transform:
               imgs = self.transform(imgs) 
        
        return imgs, labels
