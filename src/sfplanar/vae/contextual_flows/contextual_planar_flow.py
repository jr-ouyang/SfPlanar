import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualPlanarFlow(nn.Module):
    def __init__(self, latent_dim, context_dim, num_layers, reparam):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.reparam = reparam
        self.flow = nn.ModuleList([
            ContextualPlanarLayer(latent_dim, context_dim, reparam) for k in range(num_layers)
        ])
        
    def forward(self, z, context):
        log_jac_det = 0.
        for layer in self.flow:
            z, log_jac_det_i = layer(z, context)
            log_jac_det = log_jac_det + log_jac_det_i  # avoid in-place op for autograd safety
        return z, log_jac_det


class ContextualPlanarLayer(nn.Module):
    """A single layer of planar flow
        f(z) = z + v * tanh( w^T z + b )
    where 
        w^T v > -1
    
    Reparameterization
        v = v_ + [m(w^T v_) - w^T v_] w/|w|^2
    where 
        new m(x) = e^x-1 if x<0, x if x >=0 (singularity-free)
    or
        old m(x) = log(1+exp(x)) - 1 (singularity)
    """
    def __init__(self, latent_dim, context_dim, reparam):
        super().__init__()
        self.latent_dim  = latent_dim
        self.context_dim = context_dim
        self.reparam     = reparam

        self.amortized_w_net = nn.Linear(context_dim, latent_dim)
        self.amortized_v_net = nn.Linear(context_dim, latent_dim)
        self.amortized_b_net = nn.Linear(context_dim, 1)

        if reparam == 'New':
            self.compute_m = self.compute_m_new
        elif reparam == 'Old':
            self.compute_m = self.compute_m_old
        else:
            raise ValueError(f"Invalid reparam {reparam!r}. Expected 'New' or 'Old'.")
    
    def compute_amortized_params(self, context):
        # v_ is the underlying unconstrained parameter for v
        batch_size = context.shape[0]
        w  = self.amortized_w_net(context).view(batch_size, 1, self.latent_dim)
        v_ = self.amortized_v_net(context).view(batch_size, self.latent_dim, 1)
        b  = self.amortized_b_net(context).view(batch_size, 1, 1)
        return w, v_, b

    def compute_m_old(self, wv_):
        return F.softplus(wv_) - 1.  # (B,1,1)
    
    def compute_m_new(self, wv_):
        return F.elu(wv_)            # (B,1,1)
        # Equivalent forms were tested:
        # 
        # This one is slower
        #     zero = torch.zeros_like(wv_)
        #     return torch.minimum(zero, wv_).expm1() + torch.maximum(zero, wv_)
        # 
        # These two can overflow
        #     cond = wv_<0
        #     return torch.expm1(wv_) * cond + wv_ * (~cond)
        #     return torch.where(cond, torch.expm1(wv_), wv_)  
    
    def forward(self, z, context):
        batch_size = context.shape[0]
        z = z.view(batch_size, self.latent_dim, 1)
        w, v_, b = self.compute_amortized_params(context)
        
        # reparameterization
        wv_ = torch.bmm(w, v_)        # (B,1,1) <- (B,1,D) * (B,D,1)
        m_wv_ = self.compute_m(wv_)   # (B,1,1)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)       # (B,1,1)
        v = v_ + (m_wv_ - wv_) * w.transpose(2, 1) / w_norm_sq   # (B,D,1)
        
        # compute flow with reparameterized v
        wzb = torch.bmm(w, z) + b     # (B,1,1) <- (B,1,D)@(B,D,1)+(B,1,1)
        tanh_wzb = torch.tanh(wzb)    # (B,1,1)
        fz = z + v * tanh_wzb         # (B,D,1) <- (B,D,1)+(B,D,1)*(B,1,1)
        fz = fz.squeeze(2)            # (B,D)
        
        # compute log jacobian determinant
        temp = m_wv_ * (1. - tanh_wzb**2)                # (B,1,1)
        log_jac_det = torch.log1p(temp).squeeze((1,2))   # (B,)
        return fz, log_jac_det

