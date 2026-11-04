import math
import torch
import matplotlib.pyplot as plt


class TargetToy1:
    def compute_log_target(self, z):
        '''
        z: shape (*batch_shape, data_dim=2)
        Examples:    
          (10, 2)      for model training
          (10, 10, 2)  for plotting 
        Note: 
          The log-density returned is normalized.
        '''
        z_norm = (z**2).sum(axis=-1).sqrt()    # (*B,2) -> (*B,)
        decay_term  = - 0.5 * ((z_norm-2)/0.4)**2
        frac1  = (z[...,0]-2)/0.6
        frac2  = (z[...,0]+2)/0.6
        log_term = torch.logaddexp(-0.5*frac1**2, -0.5*frac2**2)
        log_const  = 1.8775015
        return decay_term + log_term - log_const     # (*B,)


class TargetToy2:
    def compute_log_target(self, z):
        decay_term = - 0.5 * (z[...,0]/5)**2         # (*B,)
        sin_term   = torch.sin(math.pi*z[...,0]/2)
        frac_term  = - 0.5 * ((z[...,1]-sin_term)/0.4)**2
        log_const   = math.log(4*math.pi)
        return decay_term + frac_term - log_const    # (*B,)



def plot_contour_and_surface3d(target, *, 
    figsize=(6, 3), 
    xlim=4, ylim=4, 
    x_points=200, y_points=200, 
    alpha=0.1, lw=0.2, 
    elev=50, azim=-80, roll=0,
):
    """
    Generates a contour plot and a 3D surface plot 
    of the 2D target log-density side by side.

    Assumes the target class implements `compute_log_target(z)` with
    z of shape (..., 2).
    """
    x = torch.linspace(-xlim, xlim, x_points)
    y = torch.linspace(-ylim, ylim, y_points)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    XY = torch.stack([X, Y], axis=2)    # (x_points, y_points, 2)
    Z = target.compute_log_target(XY).exp()
    
    fig = plt.figure(figsize=figsize, layout="constrained")        
    ax = fig.add_subplot(1, 2, 1)
    ax.contourf(X, Y, Z, cmap='coolwarm')                   
    ax.locator_params(axis='both', nbins=5)
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', alpha=alpha, lw=lw)
    ax.view_init(elev=elev, azim=azim, roll=roll)

