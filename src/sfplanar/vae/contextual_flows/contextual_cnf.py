import torch
import torch.nn as nn
import torchode


class ContextualCNF(nn.Module):
    """Continuous Normalizing Flow"""
    def __init__(self, latent_dim, context_dim, num_layers, 
                 ode_method, atol, rtol):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.amortized_w_net = nn.Linear(context_dim, num_layers*latent_dim)
        self.amortized_v_net = nn.Linear(context_dim, num_layers*latent_dim)
        self.amortized_g_net = nn.Linear(context_dim, num_layers*latent_dim)
        self.amortized_b_net = nn.Linear(context_dim, num_layers)
        self.ode_method = ode_method
        self.atol = atol
        self.rtol = rtol
        
        self.ode_func = CNFode_func()
        term = torchode.ODETerm(self.ode_func, with_args=True)
        if ode_method == 'dopri5':
            step_method = torchode.Dopri5(term=term)
        elif ode_method == 'tsit5':
            step_method = torchode.Tsit5(term=term)
        else:
            raise Exception("Please use 'Dopri5' or 'Tsit5'")
        step_size_controller = torchode.IntegralController(atol=atol, rtol=rtol, term=term)
        solver = torchode.AutoDiffAdjoint(step_method, step_size_controller)
        self.jit_solver = torch.compile(solver)

    def forward(self, z, context):
        # compute amortized flow parameters
        w = self.amortized_w_net(context).view(-1, self.num_layers, 1, self.latent_dim)
        v = self.amortized_v_net(context).view(-1, self.num_layers, self.latent_dim, 1)
        g = self.amortized_g_net(context).view(-1, self.num_layers, self.latent_dim, 1)
        b = self.amortized_b_net(context).view(-1, self.num_layers, 1, 1)
        v = v * torch.sigmoid(g)
        flow_params = (w, v, b)

        # solve the initial value problem
        logpz = torch.zeros(z.shape[0], 1).to(z.device)
        y0 = torch.cat((z, logpz), 1)
        t_eval = torch.tensor([0., 1.]).to(z.device).expand(z.shape[0], -1)
        problem = torchode.InitialValueProblem(y0=y0, t_eval=t_eval)
        sol = self.jit_solver.solve(problem, args=flow_params)
        
        # for all elements in the batch, get the final y-state
        sol_T       =   sol.ys[:,-1]  # (B,T,D+1) -> (B,D+1)
        fz          =   sol_T[:,:-1]  # (B,D) 
        log_jac_det = - sol_T[:, -1]  # (B,)  NB: log_jac_det = logp_z0 - logp_zT
        return fz, log_jac_det


class CNFode_func(nn.Module):
    """Define the ODE function for CNF
    """
    def __init__(self):
        super().__init__()

    def forward(self, t, y, flow_params):
        # unpack amortized flow parameters
        w, v, b = flow_params  # w~(B,L,1,D) v~(B,L,D,1) b~(B,L,1,1)
        num_layers = w.shape[1]
        
        # the last column is logpz, which is not used here
        z = y[:,:-1]       # (B,D)
        
        # compute dz_dt
        Z = z.unsqueeze(1).unsqueeze(3).expand(-1, num_layers, -1, -1) #(B,L,D,1)
        h = torch.tanh(torch.matmul(w,Z)+b)      # (B,L,1,D)@(B,L,D,1)->(B,L,1,1)
        dz_dt = (v * h).sum(1)      # (B,L,D,1)*(B,L,1,1)->(B,L,D,1).sum->(B,D,1)
        dz_dt = dz_dt.squeeze(2)    # (B,D)
        
        # compute dlogp_z_dt by formula
        wv = torch.matmul(w, v)     # (B,L,1,D)@(B,L,D,1) -> (B,L,1,1)
        tr_dz_dt = (1-h**2) * wv    # (B,L,1,1)*(B,L,1,1) -> (B,L,1,1)
        tr_dz_dt = tr_dz_dt.sum(1).squeeze(2)  # .sum-> (B,1,1) -> .sque-> (B,1)
        dlogpz_dt = - tr_dz_dt
        return torch.cat((dz_dt, dlogpz_dt), 1)    # (B, D+1)

