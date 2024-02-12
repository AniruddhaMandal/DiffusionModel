import numpy as np
from torch  import conv2d, nn 
import torch

class Diffusion(nn.Module):
    def __init__(self,T):
        super(Diffusion, self).__init__()
        self.time_steps = T
        self.latent_dim = 32
        self.steps_sampler()
        
        # output dim by encode_time: (,latent_dim)
        self.encode_time = nn.Sequential(
            nn.Linear(1,10),
            #nn.Linear(10,10),
            #nn.Linear(10,self.latent_dim*self.latent_dim)
            )

        # output dim by neural_net_encode: (,latent_dim)
        self.neural_net_encode = nn.Sequential(
            #(bath_size,3,300,300) -> (batch_size,16, 296,296)
            nn.Conv2d(3,16,(5,5)),

            #(batch_size,16,296,296) -> (batch_size,16,146,146)
            nn.MaxPool2d((5,5),(2,2)),

            #(batch_size,16,146,146) -> (batch_size, 32,142,142)
            nn.Conv2d(16,32,(5,5)),

            #(batch_size,32,142,104) -> (batch_size,32,69,69)
            nn.MaxPool2d((5,5),(2,2)),

            #(batch_size,32,69,69) -> (batch_size,63,64,64)
            nn.Conv2d(32,63,(6,6)),

            #(batch_size,63,64,64) -> (batch_size,63,32,32)
            nn.MaxPool2d((2,2),(2,2))

        )


        self.neural_net_decode = nn.Sequential(
        )

    def steps_sampler(self):
        self.bita = 0.99+0.009*np.random.rand(self.time_steps)
        self.alpha = 1-self.bita
        self.alpha_bar = [self.alpha[0]]
        for i in range(1,self.time_steps):
            self.alpha_bar.append(self.alpha[i]*self.alpha_bar[i-1])
        
        
    def forward(self,x: torch.Tensor,time):
        x = self.neural_net_encode(x)
        time = self.encode_time(time)
        print(f"Encoded time: {time.shape}")
        return x
        
        


    def sample(self, random_vector, label, time):
        output_vector = random_vector
        for i in range(time):
           output_vector = self.neural_net(output_vector,label, self.encode_time(time-i)) 

        return output_vector
