import torch
import torch.nn as nn


class ContextualSylvesterFlow(nn.Module):
    """Orthogonal Sylvester Flow, van den Berg et al., 2018
    
    The transformation of each layer is given by
        f(z) = z + Q R1 tanh( R2 Q^T z + b ), 
    where R1, R2 are upper triangular matrices with diag(R1*R2) > -1 
        and Q is a matrix with orthonormal column vectors
        
    Implemented as
        f(z^T) = z^T + tanh( z^T Q R2^T + b^T ) R1^T Q^T
    where
        z  ~ (B, D) -> (B, 1, D)
        r1 ~ (B, M, M, L)
        r2 ~ (B, M, M, L)
        b  ~ (B, 1, M, L)
        q_ortho ~ (L, B, D, M)
    """
    def __init__(self, latent_dim, context_dim, num_ortho_vecs, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_ortho_vecs = num_ortho_vecs
        self.num_layers = num_layers
        self.q_ortho_max_step = 100
        self.q_ortho_tol = 1e-6
        assert (num_ortho_vecs <= latent_dim) and (num_ortho_vecs > 0)
        
        # Put identity in buffer so that it will be moved to GPU if needed
        identity = torch.eye(num_ortho_vecs).unsqueeze(0)  # Add batch dimension
        self.register_buffer('_eye', identity)
        
        # Masks needed for triangular R1 and R2.
        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)    # (1, M, M, 1)
        diag_idx = torch.arange(0, num_ortho_vecs).long()
        self.register_buffer('triu_mask', triu_mask)
        self.register_buffer('diag_idx', diag_idx)
        
        # for amortized flow parameters
        self.amor_r1r2_bind_net = nn.Linear(context_dim, num_layers*num_ortho_vecs*num_ortho_vecs)
        self.amor_diag1_net = nn.Linear(context_dim, num_layers*num_ortho_vecs)
        self.amor_diag2_net = nn.Linear(context_dim, num_layers*num_ortho_vecs)
        self.amor_q_net = nn.Linear(context_dim, num_layers*latent_dim*num_ortho_vecs)
        self.amor_b_net = nn.Linear(context_dim, num_layers*num_ortho_vecs)
        
    def get_amortized_params(self, context):
        # upper triangular matrices: r1, r2
        r1r2_bind = self.amor_r1r2_bind_net(context)
        diag1 = torch.tanh(self.amor_diag1_net(context))  # tanh() so that
        diag2 = torch.tanh(self.amor_diag2_net(context))  # R1_ii * R2_ii > -1
        
        r1r2_bind = r1r2_bind.reshape(-1, self.num_ortho_vecs, self.num_ortho_vecs, self.num_layers)
        diag1 = diag1.reshape(-1, self.num_ortho_vecs, self.num_layers) # (B, M, L)
        diag2 = diag2.reshape(-1, self.num_ortho_vecs, self.num_layers) # (B, M, L)
        
        r1 = r1r2_bind * self.triu_mask                  # (B, M, M, L)
        r2 = r1r2_bind.transpose(2, 1) * self.triu_mask  # (B, M, M, L)
        
        r1[:, self.diag_idx, self.diag_idx, :] = diag1
        r2[:, self.diag_idx, self.diag_idx, :] = diag2
        
        q = self.amor_q_net(context)    # (B, L*D*M)
        b = self.amor_b_net(context)    # (B, L*M)
        b = b.reshape(-1, 1, self.num_ortho_vecs, self.num_layers) # (B,1,M,L)
        return r1, r2, q, b
        
    def forward(self, z, context):
        r1, r2, q, b = self.get_amortized_params(context)
        q_ortho = self.batch_construct_orthogonal(q)  # (L, B, D, M)
        
        log_jac_det = 0.
        zk = z.unsqueeze(1)  # (B,1,D)
        for k in range(self.num_layers):
            r1_k = r1[:, :, :, k]   # (B,M,M)
            r2_k = r2[:, :, :, k]   # (B,M,M)
            q_ortho_k = q_ortho[k, :, :, :] # (B,D,M)
            b_k = b[:, :, :, k]     # (B,1,M)
            
            # Save diagonals for log_det_j
            diag_r1 = r1_k[:, self.diag_idx, self.diag_idx]   # (B,M)
            diag_r2 = r2_k[:, self.diag_idx, self.diag_idx]   # (B,M)
            
            qr2 = torch.bmm(q_ortho_k, r2_k.transpose(2, 1))  # (B,D,M)
            qr1 = torch.bmm(q_ortho_k, r1_k)                  # (B,D,M)
            
            r2qzb = torch.bmm(zk, qr2) + b_k                  # (B,1,M)
            h_r2qzb = torch.tanh(r2qzb)                       # (B,1,M)
            zk = torch.bmm(h_r2qzb, qr1.transpose(2, 1)) + zk # (B,1,D)
            
            temp = (1-h_r2qzb**2).squeeze(1) * diag_r1 * diag_r2 # (B,M)
            log_jac_det = log_jac_det + torch.log1p(temp).sum(1) # (B,)
        
        fz = zk.squeeze(1)   # (B,D)
        return fz, log_jac_det
        
    def batch_construct_orthogonal(self, q):
        """Batch orthogonal matrix construction.
        input: q ~ (B, L*D*M)
        return: batches of orthogonalized matrices ~ (L, B, D, M)
        """
        q = q.reshape(-1, self.latent_dim * self.num_ortho_vecs)  # (B*L, D*M)
        norm = torch.linalg.norm(q, ord=2, dim=1, keepdim=True)   # (B*L, 1)
        q_ortho = torch.div(q, norm)                              # (B*L, D*M)
        q_ortho = q_ortho.reshape(-1, self.latent_dim, self.num_ortho_vecs) #(B*L, D, M)
        
        # Iterative orthogonalization
        for s in range(self.q_ortho_max_step):
            tmp = torch.bmm(q_ortho.transpose(2, 1), q_ortho)
            tmp = self._eye - tmp                  # I - Q^T Q
            tmp = self._eye + 0.5 * tmp            # I + 0.5*(I - Q^T Q)
            q_ortho = torch.bmm(q_ortho, tmp)      # Q [ I + 0.5*(I - Q^T Q) ]
            
            # Testing for convergence
            test = torch.bmm(q_ortho.transpose(2, 1), q_ortho) - self._eye
            max_norm = torch.linalg.norm(test, ord='fro', dim=(1,2)).max()
            if max_norm <= self.q_ortho_tol:
                break
        
        if max_norm > self.q_ortho_tol:
            warnings.warn(
                f'Orthogonalization not complete! Final max norm = {max_norm}'
            )
            
        q_ortho = q_ortho.reshape(-1, self.num_layers, self.latent_dim, self.num_ortho_vecs)
        q_ortho = q_ortho.transpose(0, 1)   # (L, B, D, M)
        return q_ortho

