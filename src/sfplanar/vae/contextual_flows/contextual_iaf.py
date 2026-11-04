import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualIAF(nn.Module):
    '''Inverse Autoregressive Flow
            Kingma et al., 2016, NIPS
            https://arxiv.org/abs/1606.04934
    '''
    def __init__(self, latent_dim, context_dim, hidden_dim, num_layers, activation):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation()
        self.fc_h_context = nn.Linear(context_dim, hidden_dim)
        
        # create masks for each flow layer
        degrees = self.create_degrees()
        masks = self.create_masks(degrees)
        
        # build flow model
        flow = [IAFblock(masks[k], activation) for k in range(num_layers)]
        self.flow = nn.ModuleList(flow)
        
    def forward(self, z, context):
        h_context = self.activation(self.fc_h_context(context))
        log_jac_det = 0.
        for layer in self.flow:
            z, log_jac_det_i = layer(z, h_context)
            log_jac_det = log_jac_det + log_jac_det_i  # avoid in-place op
        return z, log_jac_det
        
    def create_degrees(self):
        # create degrees for each flow layer
        degrees = []
        for k in range(self.num_layers):
            # alternating degrees for input units
            if k % 2 == 0:
                degrees_k_0 = torch.arange(1, self.latent_dim + 1)
            else:
                degrees_k_0 = torch.arange(self.latent_dim, 0, -1)
            
            # sequential degrees for the hidden units (assume latent_dim >=2)
            degrees_k_1 = torch.arange(self.hidden_dim) % (self.latent_dim - 1) + 1
            degrees_k_2 = torch.arange(self.hidden_dim) % (self.latent_dim - 1) + 1
            
            # save degrees of the k-th flow layer
            degrees_k = [degrees_k_0, degrees_k_1, degrees_k_2]
            degrees.append(degrees_k)
        return degrees
        
    def create_masks(self, degrees):
        # create masks for each flow layer
        dtype = torch.get_default_dtype()
        masks = []
        for k in range(self.num_layers):
            degrees_k = degrees[k]
            mask_0 = degrees_k[0] <= degrees_k[1][:, None] # input layer
            mask_1 = degrees_k[1] <= degrees_k[2][:, None] # hidden layer
            mask_2 = degrees_k[0][:, None] > degrees_k[2]  # output layer
            masks_k = [mask_0.to(dtype), mask_1.to(dtype), mask_2.to(dtype)]
            masks.append(masks_k)
        return masks


class IAFblock(nn.Module):
    '''Building Block of Inverse Autoregressive Flow'''
    def __init__(self, masks, activation):
        super().__init__()
        self.made = MADE(masks, activation)
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
        
    def forward(self, z, h_context):
        m, s = self.made(z, h_context)
        sigma = self.sigmoid(s)
        fz = sigma * z + (1-sigma) * m
        log_jac_det = self.logsigmoid(s).sum(-1)    # log(sigma).sum(-1)
        return fz, log_jac_det


class MADE(nn.Module):
    '''Masked Autoencoder Distribution Estimator (MADE) with Gaussian conditionals
        Germain et al., 2015, ICML
        MADE: Masked Autoencoder for Distribution Estimation
        https://arxiv.org/abs/1502.03509
    '''
    def __init__(self, masks, activation):
        super().__init__()
        self.activation = activation()
        self.masked_linear_1 = MaskedLinear(masks[0])
        self.masked_linear_2 = MaskedLinear(masks[1])
        self.masked_linear_m = MaskedLinear(masks[2])
        self.masked_linear_s = MaskedLinear(masks[2])
    
    def forward(self, z, h_context):
        h1 = self.activation(self.masked_linear_1(z)) + h_context
        h2 = self.activation(self.masked_linear_2(h1))
        m = self.masked_linear_m(h2)
        s = self.masked_linear_s(h2)
        return m, s


class MaskedLinear(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.out_features = mask.shape[0]
        self.in_features  = mask.shape[1]
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features)))
        self.bias   = nn.Parameter(torch.empty( self.out_features ))
        self.reset_parameters()
        self.register_buffer("mask", mask)

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias,   -bound, bound)

    def forward(self, z):
        return F.linear(z, self.mask * self.weight, self.bias)

