import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..special import exp_linear


class PlanarFlow(nn.Module):
    def __init__(self, dim, num_layers, reparam, linear_layer=True, replicate=None):
        super().__init__()
        self.dim          = dim
        self.num_layers   = num_layers
        self.reparam      = reparam
        self.linear_layer = linear_layer
        self.replicate    = replicate
        self.flow = nn.ModuleList([
            *( [LowerTriLinearLayer(dim, replicate=replicate)] if linear_layer else [] ),
            *[
                PlanarLayer(dim, reparam=reparam, replicate=replicate, layer=k) for k in range(num_layers)
            ]
        ])

    def forward(self, z):
        log_jac_det = 0.
        for layer in self.flow:
            z, log_jac_det_i = layer(z)
            log_jac_det = log_jac_det + log_jac_det_i
        return z, log_jac_det

    @torch.no_grad()
    def inverse(self, z):
        was_training = self.training
        self.eval()
        log_jac_det = 0.
        for layer in reversed(self.flow):
            z, log_jac_det_i = layer.inverse(z)
            log_jac_det = log_jac_det + log_jac_det_i
        self.train(was_training)
        return z, log_jac_det
    

class PlanarLayer(nn.Module):
    """A single layer of planar flow
        f(z) = z + v * tanh( w^T z + b )
    where 
        w^T v > -1
        Reparameterization is defined in the child classes

    Reparameterization
        v = v_ + [m(w^T v_) - w^T v_] w/|w|^2
    where 
        new m(x) = e^x-1 if x<0, x if x >=0 (singularity-free)
    or
        old m(x) = log(1+exp(x)) - 1 (singularity)
    """
    def __init__(self, dim, *, reparam, replicate, layer, init_params=None):
        super().__init__()
        self.dim = dim
        self.reparam = reparam
        self.replicate = replicate
        self.layer = layer

        if reparam == 'New':
            self.compute_v_and_wv = self._compute_v_and_wv_new
        elif reparam == 'Old':
            self.compute_v_and_wv = self._compute_v_and_wv_old
        else:
            raise ValueError(f"Invalid reparam {reparam!r}. Expected 'New' or 'Old'.")

        if init_params is None:
            init_params = self.generate_init_params(reparam, replicate, layer)
        self.init_params = init_params

        self.w = nn.Parameter(init_params["w"])
        self.b = nn.Parameter(init_params["b"])
        self.v_ = nn.Parameter(init_params["v_"])

    def generate_init_params(self, reparam, replicate, layer):
        if replicate is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng([replicate, layer+1])

        bound = 1.0 / self.dim**0.5
        w = rng.uniform(-bound, bound, size=self.dim)
        b = rng.uniform(-bound, bound, size=1)
        v = rng.uniform(-bound, bound, size=self.dim)

        # transform the effective parameter `v` to the underlying parameter `v_`
        #     v_  = v - [m(w^T v_) - w^T v_] w/|w|^2
        # where
        #     m(w^T v_) = w^T v  and  w^T v_  = m^(-1)(w^T v)
        wv = np.dot(w, v)
        if reparam == 'New':
            # wv_ = np.log1p(wv) * (wv < 0) + wv * (wv >= 0)
            if wv < 0:
                wv_ = np.log1p(wv)
            else:
                wv_ = wv
        else:
            wv_ = np.log(np.expm1(wv+1))  # y=log(1+e^x)-1 -> x=log(exp(y+1)-1)

        v_ = v - (wv - wv_) * w/np.dot(w, w)

        dtype=torch.get_default_dtype()
        return {
            "w": torch.tensor(w, dtype=dtype),
            "b": torch.tensor(b, dtype=dtype),
            "v_": torch.tensor(v_, dtype=dtype)
        }

    def _compute_v_and_wv_new(self):
        wv_ = torch.dot(self.w, self.v_)
        if wv_ < 0:
            wv = torch.expm1(wv_)
            v = self.v_ + (wv - wv_) * self.w/torch.dot(self.w, self.w)
        else:
            wv = wv_
            v = self.v_
        return v, wv

    def _compute_v_and_wv_old(self):
        # we use `softplus(x)` rather than `log1p(exp(x)) - 1` to compute wv, 
        #    - avoid overflow
        #    - also fewer steps in backward/forward pass -> faster
        wv_ = torch.dot(self.w, self.v_)
        wv = F.softplus(wv_) - 1.0
        v = self.v_ + (wv - wv_) * self.w/torch.dot(self.w, self.w)
        return v, wv

    def forward(self, z):
        v, wv = self.compute_v_and_wv()
        hyper_level = z@self.w + self.b      # (n,)    <- (n,dim)@(dim,) + (1,)
        tanh = torch.tanh(hyper_level)       # (n,)
        fz = z + tanh.unsqueeze(dim=1) * v   # (n,dim) <- (n,dim) + (n,1)*(dim,)
        log_jac_det = torch.log1p(wv * (1-tanh**2))
        return fz, log_jac_det

    def inverse(self, target, tol=None, max_iter=100, return_stats=False):
        '''find z such that f(z)=target'''
        if tol is None:
            dtype = torch.get_default_dtype()
            if dtype == torch.float64:
                tol = 1e-9    # tol ~ 5e6 × machine eps, which is ~ 2.2e-16
            elif dtype == torch.float32:
                tol = 6e-6    # tol ~  50 × machine eps, which is ~ 1.2e-07
            else:
                # float16:  eps ~ 9.77e-4, tol ~ 0.03125
                # bfloat16: eps ~ 7.81e-3, tol ~ 0.08839
                tol = torch.finfo(dtype).eps**0.5

        # adjust dimension 
        w = self.w.unsqueeze(1)                      # (d, 1)
        v = self.compute_v_and_wv()[0].unsqueeze(1)  # (d, 1)
        b = self.b                                   # (1,)
        
        # precompute terms that do not depend on z
        v_wT = torch.mm(v, w.T)        # (d, d)
        wT_v = torch.mm(w.T, v)        # (1, 1)
        I = torch.eye(w.shape[0], device=w.device, dtype=w.dtype) # (d, d)
        
        # newton's method
        z = target.clone()             # use target itself as the initial point
        for iter in range(max_iter):
            fz, log_jac_det = self(z)  # (n, d) and (n,)
            residual = target - fz     # (n, d)
            res_norm = torch.linalg.norm(residual, ord=float('inf'), dim=1)

            if torch.all(res_norm < tol):
                break

            z_col = z.unsqueeze(2)     # (n, d, 1)
            wT_z_b = w.T @ z_col + b   # (n, 1, 1)
            f_prime_inv = I - v_wT / (torch.cosh(wT_z_b)**2 + wT_v) # (n, d, d)
            delta = torch.bmm(f_prime_inv, residual.unsqueeze(2))   # (n, d, 1)
            z = z + delta.squeeze(2)   # (n, d)
        else:
            print(f"Newton's method did not converge within {max_iter} iterations")

        if return_stats:
            return z, log_jac_det, res_norm, tol, iter
        else:
            return z, log_jac_det


class LowerTriLinearLayer(nn.Module):
    """A linear layer with lower triangular weight matrix"""
    def __init__(self, dim, *, replicate, init_params=None):
        super().__init__()
        self.dim = dim

        if init_params is None:
            init_params = self.generate_init_params(replicate)
        self.init_params = init_params
        
        self.W_ = nn.Parameter(init_params["W_"])
        self.b = nn.Parameter(init_params["b"])

    def generate_init_params(self, replicate):
        if replicate is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng([replicate])

        bound = 1.0 / self.dim**0.5
        W_ = rng.uniform(-bound, bound, size=(self.dim, self.dim))
        W_ = np.tril(W_)
        b = rng.uniform(-bound, bound, size=self.dim)

        dtype=torch.get_default_dtype()
        return {
            "W_": torch.tensor(W_, dtype=dtype),
            "b": torch.tensor(b, dtype=dtype)
        }
    
    @property
    def W_and_log_jac_det(self):
        # We reparameterize the diagnoal with `exp_linear()`:
        #     exp(x) if x < 0
        #     x + 1  if x >= 0
        # At x = 0, both branches evaluate to 1 (continuously differentiable at 0). 
        W_diag = exp_linear(self.W_.diag())
        W = torch.tril(self.W_, -1) + torch.diag(W_diag)
        log_jac_det = torch.sum(torch.log(W_diag))
        return W, log_jac_det

    def forward(self, z):
        W, log_jac_det = self.W_and_log_jac_det
        fz = F.linear(z, W, self.b)
        return fz, log_jac_det.expand(z.size(0))
        
    def inverse(self, fz):
        W, log_jac_det = self.W_and_log_jac_det
        rhs = (fz-self.b).T              # (dim, n)
        zT = torch.linalg.solve(W, rhs)  # (dim, n)
        return zT.T, log_jac_det.expand(fz.size(0))

