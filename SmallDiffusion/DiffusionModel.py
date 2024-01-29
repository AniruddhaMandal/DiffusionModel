from torch  import nn 


class Diffusion(nn.Module):
    def __init__(self):
        self.encode_time = nn.Sequential(
            nn.Linear(1,10),
            nn.Linear(10,10))

        self.neural_net = nn.Sequential(

        )
    def forward(self):
        pass 

         
    def sample(self, random_vector, label, time):
        output_vector = random_vector
        for i in range(time):
           output_vector = self.neural_net(output_vector,label, self.encode_time(time-i)) 

        return output_vector