import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class CNN(torch.nn.Module):
    def __init__(self):
        super.__init__()
        # ConvNet architecture
        # self.layers = torch.nn.Sequential(
        #     # TODO: Add network layers
        # )

        self.input_height = 82 #input_height
        self.input_width = 4 #input_width
        self.n_classes = 3 #0 (neither), 1(donor), 2 (acceptor) n_classes
        self.conv1 = nn.Conv2d(1,8,3,1) #(1, 32, 3, 1)   #(1,8,3,1)     
        self.conv2 = nn.Conv2d(8,16,3,2)#(32, 64, 3, 2)   #(8,16,3,2)
        self.pool1 = nn.AvgPool2d(kernel_size=12)    #()
        self.conv3 = nn.Conv2d(16,6,1,1)#(64, 6, 1, 1) #(16,6,1,1)
        # raise NotImplementedError()
   
    def forward(self,x):

        #reshape vectors to images
        elems = torch.numel(x)
        batch = int(elems/ (self.input_height*self.input_width))
        x = np.reshape(x, (batch, self.input_height,self.input_width))
        x = torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = torch.reshape(x,(batch,self.n_classes))

        return x
    
