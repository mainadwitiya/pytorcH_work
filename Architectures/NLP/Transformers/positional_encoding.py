import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=2500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) #initialize a tensor filled with scaler values of zero
        
        position = torch.arange(0, max_len).unsqueeze(1) #torch.unsqueeze adds an additional dimension to the tensor.
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #division term of the formula of PE
        print(div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # buffers are used when you need to save the parameters in which will be restored from state_dict but not needed to be trained by optimizers
        
    def forward(self, x):
        '''
        Tensors and variables as same. Variables are  wrappers for the tensors so you can now easily auto compute the gradients.
        So if a tensor was batman…
        A Variable would be batman but with his utility belt on
        '''
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    

%matplotlib inline
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20,0)
y = pe.forward(Variable(torch.zeros(1, 100, 20)))
plt.plot(np.arange(100), y[0, :, 3:8].data.numpy())
plt.legend(["dim %d"%p for p in [3,4,5,6,7]])
