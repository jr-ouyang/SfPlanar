import time

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import format_runtime


class TargetBayesianRegression:
    def __init__(self, raw_data, likelihood, prior):
        self.model_data = likelihood.process_data(raw_data)
        self.likelihood = likelihood
        self.prior = prior
        
    def compute_log_target(self, beta):
        # the target is the unnormalized posterior distribution
        log_likelihood = self.likelihood.compute_log_likelihood(beta, self.model_data)
        log_prior = self.prior.compute_log_prior(beta)
        return log_likelihood + log_prior


class LinearRegressionLikelihood:
    def __init__(self, sigma=1.0):
        self.sigma = sigma    # Error ~ N(0,1) is assumed known
        self.sigma2 = sigma**2

    def compute_log_likelihood(self, beta, model_data):
        # beta: shape (*B, p)
        b  = beta.unsqueeze(-1)      # (*B, p, 1)
        bT = b.transpose(-2, -1)     # (*B, 1, p)

        yTy = model_data['yTy']      # scalar
        yTX = model_data['yTX']      # (p,)
        XTX = model_data['XTX']      # (p, p)
        n = model_data['n']

        pi = torch.tensor(torch.pi, device=beta.device)
        const = n * torch.log(2 * pi * self.sigma2)

        # (p,) @ (*B, p, 1) -> (*B, 1) -> (*B,)
        yTXb = (yTX @ b).squeeze(-1)

        # (*B, 1, p) @ (p, p) @ (*B, p, 1) -> (*B, 1, 1) -> (*B,)
        bTXTXb = (bT @ XTX @ b).squeeze(-1, -2)

        rss = yTy - 2 * yTXb + bTXTXb    # (*B,)
        log_likelihood = - 0.5 * (const + rss / self.sigma2)   # (*B,)
        return log_likelihood
    
    def process_data(self, raw_data):
        y = raw_data["y"]             # (n,)
        X = raw_data["X"]             # (n, p)

        model_data = {
            "yTy": torch.dot(y, y),   # scalar
            "yTX": y @ X,             # (p,)
            "XTX": X.T @ X,           # (p, p)
            "n":   X.shape[0],        # scalar
        }
        return model_data


class LogisticRegressionLikelihood:
    def compute_log_likelihood(self, beta, model_data):
        # beta: shape (*B, p)
        b = beta.unsqueeze(-1)      # (*B, p, 1)

        y = model_data['y']         # (n,)
        X = model_data['X']         # (n, p)
        
        # (n, p) @ (*B, p, 1) -> (*B, n, 1) -> (*B, n)
        logits = (X @ b).squeeze(-1)

        # reshape y to (1,...1, n) then expand to (*B, n) to match logits
        batch_ndim = beta.ndim - 1
        target = y.reshape((1,) * batch_ndim + (-1,)).expand_as(logits)

        cross_entropy = F.binary_cross_entropy_with_logits(
            input = logits, 
            target = target, 
            reduction = 'none'
        ).sum(-1)                   # (*B, n).sum(-1) -> (*B,)
        
        log_likelihood = - cross_entropy
        return log_likelihood
    
    def process_data(self, raw_data):
        return raw_data    # no pre-processing needed for logistic regression


class UniformCauchyPrior:
    '''
    We define the uniform-Cauchy prior as a compound probability distribution:
        beta_j | gamma_j ~ Cauchy(0, gamma_j) and gamma_j ~ Unif(0, s)
    The marginal density of beta_j is
        p(beta_j) = 1/(2 pi s) log[1 + (s/beta_j)^2]
    where s > 0 is a scale parameter. 
    '''
    def __init__(self, beta_dim, scale):
        self.beta_dim = beta_dim
        self.scale = scale
    
    def compute_log_prior(self, beta):
        '''
        beta: shape (*B, beta_dim)
        log p(beta_j) = - log(2 pi s) + log{log[1 + (s/beta_j)^2]}
        log p(beta) = - p log(2 pi s) + sum(log{log[1 + (s/beta_j)^2]})
        '''
        p = self.beta_dim
        s = self.scale

        pi = torch.tensor(torch.pi, device=beta.device)
        log_const = - p * torch.log(2 * pi * s)

        log_kernel = torch.log(torch.log1p((s/beta)**2)).sum(-1) # (*B,p)->(*B,)
        log_prior = log_const + log_kernel
        return log_prior


class DGP:
    """Data Generating Process for the regression model:
    True beta:
        beta_1, beta_2 ~ Unif(-1,1)
        beta_j = 0 for j=3,...,p
    Linear regression:
        epsilon_i ~ N(0, sigma^2)
        y = X * beta + epsilon
    Logistic regression:
        p_i = 1/(1+exp(-X_i * beta))
        y_i ~ Bernoulli(p_i)
    """
    def _generate_beta_true(self, beta_dim, beta_nnz, rng):
        beta_true = np.zeros(beta_dim)
        beta_true[:beta_nnz] = rng.uniform(low=-1.0, high=1.0, size=beta_nnz)
        return beta_true

    def _generate_X(self, sample_size, beta_dim, rho, rng):
        n = sample_size
        p = beta_dim
        cov_X = np.array(
            [rho**np.abs(np.arange(p)-j) for j in range(p)]
        )
        X = rng.multivariate_normal(np.zeros(p), cov_X, size=n)
        return X
    
    def generate_data_linear(self, sample_size, beta_dim, beta_nnz, sigma, rho, replicate):
        rng = np.random.default_rng([replicate+1, replicate+2, replicate+3])

        beta_true = self._generate_beta_true(beta_dim, beta_nnz, rng)
        X = self._generate_X(sample_size, beta_dim, rho, rng)
        epsilon = rng.normal(loc=0.0, scale=sigma, size=sample_size)
        y = X @ beta_true + epsilon

        dtype = torch.get_default_dtype()
        
        raw_data = {
            "y": torch.as_tensor(y, dtype=dtype),
            "X": torch.as_tensor(X, dtype=dtype),
        }
        beta_true = torch.as_tensor(beta_true, dtype=dtype)
        return raw_data, beta_true

    def generate_data_logistic(self, sample_size, beta_dim, beta_nnz, rho, replicate):
        rng = np.random.default_rng([replicate+1, replicate+2, replicate+3])

        beta_true = self._generate_beta_true(beta_dim, beta_nnz, rng)
        X = self._generate_X(sample_size, beta_dim, rho, rng)
        logits = X @ beta_true
        prob = 1 / (1 + np.exp(-logits))
        y = rng.binomial(n=1, p=prob)

        dtype = torch.get_default_dtype()

        raw_data = {
            "y": torch.as_tensor(y, dtype=dtype), # target y should be float for BCEWithLogitsLoss
            "X": torch.as_tensor(X, dtype=dtype),
        }
        beta_true = torch.as_tensor(beta_true, dtype=dtype)
        return raw_data, beta_true


@torch.no_grad()
def estimate_log_evidence(flow, target, IS_size, *, num_batches, seed=31415926, verbose=True): 
    """
    The target distribution is
        p(beta | y, X) = p(y | beta, X) * p(beta) / p(y | X)
        posterior        likelihood       prior     model evidence
    
    For Variational Bayes to work, we only need the unnormalized target density, 
        p(y, beta | X) = p(y | beta, X) * p(beta).  
    
    But to compute KL, we also need the model evidence, p(y | X).
    
    KL(q(beta) || p(beta | y, X)) = E_q[log q(beta) - log p(beta, y | X)] + log p(y | X)
                                  = - ELBO + log p(y | X)
    
    This method estimates the model evidence in log scale:
        log p(y | X)
        = log[ 1/S sum_{s=1}^S { p(y, beta_s | X) / phi(beta_s) } ] 
        = logsumexp( log p(y, beta_s | X) - log phi(beta_s) ) - log S
    where beta_s ~ phi(beta).

    We take phi to be the trained variational distribution: phi(beta) = q(beta).

    To get the estimate, we need to compute 
        log p(y, beta_s | X) - log phi(beta_s) 
    for a large S. We do this in batches with batch_size = S / num_batches. 
    """
    time_start = time.perf_counter()

    was_training = flow.training
    flow.eval()
    device = next(flow.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    pi = torch.tensor(torch.pi, device=device)

    # check if sample size can split evenly
    batch_size, remainder = divmod(IS_size, num_batches)
    if remainder > 0:
        raise ValueError(f"Importance sampling size {IS_size} must be divisible by num_batches {num_batches}.")
    
    # compute log p(y, beta_s | X) - log q(beta_s), for log_evidence
    # compute log p(y, beta_s | X) - E[log q(beta_s)], for -ELBO
    log_joint_minus_log_q = torch.zeros(num_batches, batch_size, device=device)
    log_joint_minus_Elog_q = torch.zeros(num_batches, batch_size, device=device)
    for j_batch in range(num_batches): 
        # sample batch_size points from q(), i.e., the trained flow distribution
        z0 = torch.randn(batch_size, flow.dim, generator=rng, device=device)
        zk, log_jac_det = flow(z0)     # (batch_size,p), (batch_size,)
        beta = zk                      # (batch_size,p)

        # compute log p(y, beta_s | X), the unnormalized target density
        log_joint_beta = target.compute_log_target(beta)                 # (batch_size,)

        # compute log q(beta_s), for log_evidence
        log_q0_z0 = torch.sum(- 0.5 * (torch.log(2*pi) + z0**2), dim=-1) # (batch_size,)
        log_qk_beta = log_q0_z0 - log_jac_det                            # (batch_size,)        
        log_joint_minus_log_q[j_batch] = log_joint_beta - log_qk_beta    # (batch_size,)

        # compute E[log q(beta_s)], for -ELBO
        Elog_q0_z0 = - 0.5 * flow.dim * (torch.log(2*pi) + 1)            # scalar
        Elog_qk_beta = Elog_q0_z0 - log_jac_det                          # (batch_size,)
        log_joint_minus_Elog_q[j_batch] = log_joint_beta - Elog_qk_beta  # (batch_size,)
    
    # compute log_evidence, ELBO, and KL
    log_S = torch.log(torch.tensor(IS_size, device=device))
    log_evidence = torch.logsumexp(log_joint_minus_log_q, dim=(0,1)) - log_S # scalar
    ELBO = torch.mean(log_joint_minus_Elog_q)
    KL = log_evidence - ELBO

    flow.train(was_training)
    runtime = time.perf_counter() - time_start
    runtime_str = format_runtime(runtime)
    if verbose:
        lines = [
            f"Completed estimating log-evidence using importance sampling",
            f"    Result", 
            f"        log_evidence = {log_evidence:.6g}", 
            f"        ELBO = {ELBO:.6g}",
            f"        KL = {KL:.6g}",
            f"    Parameters",
            f"        IS_size: {IS_size:,}",
            f"        num_batches: {num_batches:,}",
            f"        batch_size: {batch_size:,}",
            f"    Runtime", 
            f"        {runtime_str}",
        ]
        print("\n".join(lines) + "\n\n")

    estimation_result = {
        "log_evidence": log_evidence,
        "ELBO": ELBO,
        "KL": KL,
        "IS_size": IS_size,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "runtime": runtime,
        "runtime_str": runtime_str,
    }

    return estimation_result

