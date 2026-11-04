import torch
import torch.nn as nn


def exp_linear(x):
    '''
    Exponential Linear Function
        exp(x) if x < 0
        x + 1  if x >= 0
    At x = 0, both branches evaluate to 1 (continuous at 0). 
    
    Mathematically, this function is continuously differentiable everywhere. 
    Implementation-wise, 
        d(min(0,x))/dx = 0.5 at x=0
        d(max(0,x))/dx = 0.5 at x=0
        f'(0) = exp(0)*0.5 + 0.5 = 1
    which is consistent with the mathematical definition.
    '''
    zero = torch.zeros_like(x)
    return torch.exp(torch.minimum(zero, x)) + torch.maximum(zero, x)
    # Equivalent forms were tested, but they can overflow
    #     cond = x<0
    #     return torch.exp(x) * cond + (x+1) * (~cond)
    #     return torch.where(cond, torch.exp(x), x+1)  


class ExpLinear(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return exp_linear(x)

