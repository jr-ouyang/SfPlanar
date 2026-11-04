from time import perf_counter
from copy import deepcopy

import torch
import torch.nn as nn

from ..utils import format_runtime


class FlowVBayes(nn.Module):
    """
    Variational Bayes with a Normalizing Flow.

    `FlowVBayes` takes a flow model and trains it to approximate a given target distribution.
        - `flow` a flow model to be trained
        - `target` provides the target log-density (possibly unnormalized):
    
    Training minimizes the KL divergence (possibly unnormalized) using Monte Carlo samples 
    generated on the fly.
    """
    def __init__(self, flow, target, seed_train = 11235813):
        super().__init__()
        self.flow = flow
        self.target = target
        self.dim = flow.dim
        self.seed_train = seed_train
        self.rng_train = torch.Generator().manual_seed(seed_train)
        self.iteration = 0

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        # Move model as usual
        model = super().to(*args, **kwargs)
        # Update RNG to new device
        model.rng_train = torch.Generator(device=model.device).manual_seed(model.seed_train)
        return model

    def forward(self, z):
        zk, log_jac_det = self.flow(z)
        return zk, log_jac_det

    def loss_fn(self, zk, log_jac_det):
        pi = torch.tensor(torch.pi, device=self.device)
        # q_u(u) ~ MVN(0,I) 
        # => E[log q_u(u)] = d * E[ -0.5*log(2 pi) - u^2 /2 ] = -0.5 * d * [log(2 pi)+1]
        E_log_q0 = - 0.5 * self.dim * (torch.log(2*pi)+1)  #         scalar
        E_log_jac_det = torch.mean(log_jac_det)            # (B,) -> scalar
        E_log_qz = E_log_q0 - E_log_jac_det                #         scalar
        log_target = self.target.compute_log_target(zk)    # (B,)
        E_log_target = torch.mean(log_target)              # (B,) -> scalar
        kl = E_log_qz - E_log_target
        return kl

    def _train_loop(self, num_updates_per_loop, batch_size, optimizer):
        self.train()
        for i_update in range(num_updates_per_loop):
            z_shape = (batch_size, self.dim)
            z = torch.randn(z_shape, generator=self.rng_train, device=self.device)
            zk, log_jac_det = self(z)
            loss = self.loss_fn(zk, log_jac_det)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.iteration += 1
    
    @torch.no_grad()
    def _test_loop(self, num_batches, batch_size, seed_test):
        self.eval()

        # re-seed every time validation runs -> same data for testing
        rng_test = torch.Generator(device=self.device).manual_seed(seed_test)

        test_loss = torch.zeros(num_batches)
        for i_batch in range(num_batches):
            z_shape = (batch_size, self.dim)
            z = torch.randn(z_shape, generator=rng_test, device=self.device)
            zk, log_jac_det = self(z)
            test_loss[i_batch] = self.loss_fn(zk, log_jac_det)
        return torch.mean(test_loss)

    def train_and_test(self, num_updates, 
                       num_updates_per_loop, batch_size, 
                       test_size, batch_size_test, *, 
                       lr_init=1e-3, lr_gamma=0.95,
                       seed_test=31415926, log_every=1):
        # Check divisibility
        num_loops, r_train = divmod(num_updates, num_updates_per_loop)
        if r_train > 0:
            raise ValueError(f"num_updates {num_updates} must be divisible by num_updates_per_loop {num_updates_per_loop}.")
        num_batches_test, r_test = divmod(test_size, batch_size_test)
        if r_test > 0:
            raise ValueError(f"test_size {test_size} must be divisible by batch_size_test {batch_size_test}.")
        
        # Set up optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_init)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

        # Initial test before training
        t_init = perf_counter()
        t_delta = perf_counter()
        test_loss = self._test_loop(num_batches_test, batch_size_test, seed_test)
        test_loss_history = [test_loss]

        widths = [6, 8, 8, 7, 7]
        if log_every > 0:
            delta_time_str = format_runtime(perf_counter()-t_delta, digits=0)
            total_time_str = format_runtime(perf_counter()-t_init, digits=0)
            print(f"{'Loop':>{widths[0]}}   {'Iters':>{widths[1]}}   "
                  f"{'TestLoss':>{widths[2]}}   {'Delta':>{widths[3]}}   "
                  f"{'Total':>{widths[4]}}")
            print(f"{'':>{widths[0]}}   {'Initial':>{widths[1]}}   "
                  f"{test_loss:>{widths[2]}.3f}   {delta_time_str:>{widths[3]}}   "
                  f"{total_time_str:>{widths[4]}}")
            t_delta = perf_counter()
        
        # Start training and testing loops
        for i_loop in range(1, num_loops+1):
            self._train_loop(num_updates_per_loop, batch_size, optimizer) 
            test_loss = self._test_loop(num_batches_test, batch_size_test, seed_test)
            test_loss_history.append(test_loss) 
            scheduler.step()
            
            if log_every > 0 and (i_loop % log_every == 0):
                delta_time_str = format_runtime(perf_counter()-t_delta, digits=0)
                total_time_str = format_runtime(perf_counter()-t_init, digits=0)
                print(f"{i_loop:>{widths[0]}}   {self.iteration:>{widths[1]}}   "
                      f"{test_loss:>{widths[2]}.3f}   {delta_time_str:>{widths[3]}}   "
                      f"{total_time_str:>{widths[4]}}")
                t_delta = perf_counter()
        
        
        # Training completed: switch back to training mode and print time usage
        self.train()
        runtime = perf_counter() - t_init
        runtime_str = format_runtime(runtime, digits=0)
        iters_per_sec = self.iteration/runtime
        print(f'Time for training {num_loops} loops ({self.iteration} iters):  '
              f'{runtime_str}  @  {iters_per_sec:.2f} iter/sec\n\n')
        
        return {
            'state_dict': deepcopy(self.state_dict()),
            'test_loss_hist': torch.stack(test_loss_history) # (num_loops+1,)
        }

