from re import I
import numpy as np
from torch  import nn 


class Diffusion(nn.Module):
    def __init__(self,T):
        self.time_steps = T
        self.steps_sampler()
        self.encode_time = nn.Sequential(
            nn.Linear(1,10),
            nn.Linear(10,10))

        self.neural_net = nn.Sequential(
        )

    def steps_sampler(self):
        self.bita = 0.99+0.009*np.random.rand(self.time_steps)
        self.alpha = 1-self.bita
        self.alpha_bar = [self.alpha[0]]
        for i in range(1,self.time_steps):
            self.alpha_bar[i].append(self.alpha[i]*self.alpha_bar[i-1])
        
        
    def forward(self):
        pass 

         
    def sample(self, random_vector, label, time):
        output_vector = random_vector
        for i in range(time):
           output_vector = self.neural_net(output_vector,label, self.encode_time(time-i)) 

        return output_vector