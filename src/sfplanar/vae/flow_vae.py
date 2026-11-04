import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from torch.utils.data import DataLoader
from time import time
from copy import deepcopy

from ..special import ExpLinear
from ..utils import format_runtime


class FlowVAE(nn.Module):
    def __init__(self, num_channels, flow, seed_train=31415926):
        super().__init__()
        self.latent_dim = flow.latent_dim
        self.context_dim = flow.context_dim
        self.C1 = num_channels[1]
        self.seed_train = seed_train
        self.rng_train = torch.Generator().manual_seed(seed_train)
        self.active_rng = self.rng_train
        self.num_iters_anneal = None
        self.anneal_mode = False
        self.iteration = 0
        
        # encoder
        C0, C1 = num_channels
        self.conv = nn.Sequential(
            # in_channels  = 1 <- grayscale image
            # out_channels = C0
            # kernel_size  = 3   (stride=1 by default)
            # max pooling with kernel_size=2 (stride=kernel_size by default)
            nn.Conv2d(1,  C0, 3, padding='same'), # -> (B, C0, 28, 28)
            nn.ReLU(), 
            nn.MaxPool2d(2),                      # -> (B, C0, 14, 14)
            nn.Conv2d(C0, C1, 3, padding='same'), # -> (B, C1, 14, 14)
            nn.ReLU(), 
            nn.MaxPool2d(2)                       # -> (B, C1, 7,  7)
        )
        
        # fully connected networks for z0_mu, z0_sd, and context vector for flow parameters
        conv_out_flattened_dim = C1*49
        self.fc_mu      = nn.Linear(conv_out_flattened_dim, flow.latent_dim)
        self.fc_sd      = nn.Linear(conv_out_flattened_dim, flow.latent_dim)
        self.fc_context = nn.Linear(conv_out_flattened_dim, flow.context_dim)
        
        # activation functions
        self.exp_linear = ExpLinear()  # for fc_sd; linear for positive x to avoid overflow
        self.tanh = nn.Tanh()     # for fc_context; to avoid numerically large context
        
        # flow
        self.flow = flow
        
        # decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(flow.latent_dim, conv_out_flattened_dim),   # view()-> (B, C1, 7, 7)
            nn.ReLU()
        )
        self.convTrans = nn.Sequential(
            nn.ConvTranspose2d(C1, C0, 3, stride=2,
                               padding=1, output_padding=1), #-> (B, C0, 14, 14) 
            nn.ReLU(),
            nn.ConvTranspose2d(C0,  1, 3, stride=2, 
                               padding=1, output_padding=1)  #-> (B, 1, 28, 28)
        )

    @property
    def device(self):
        return next(self.parameters()).device
    
    def to(self, *args, **kwargs):
        # Move model as usual
        model = super().to(*args, **kwargs)
        # Update RNG to new device
        model.rng_train = torch.Generator(device=model.device).manual_seed(model.seed_train)
        return model
    
    def encode(self, x):
        conv_out = self.conv(x)
        conv_out_flattened = conv_out.view(x.size(0), -1)
        mu = self.fc_mu(conv_out_flattened)
        sd = self.exp_linear(self.fc_sd(conv_out_flattened))
        context = self.tanh(self.fc_context(conv_out_flattened))
        return mu, sd, context
    
    def draw_from_flow(self, mu, sd, context):
        eps = torch.randn(sd.size(), generator=self.active_rng, device=self.device)
        z0 = mu + sd * eps
        zk, log_jac_det = self.flow(z0, context)
        return zk, log_jac_det, z0
    
    def decode(self, zk):
        hidden = self.fc_decoder(zk)
        hidden_view = hidden.view(-1, self.C1, 7, 7)
        logits = self.convTrans(hidden_view)
        return logits
    
    def forward(self, x):
        mu, sd, context = self.encode(x)
        zk, log_jac_det, z0 = self.draw_from_flow(mu, sd, context)
        logits = self.decode(zk)
        return logits, zk, log_jac_det, z0, mu, sd
    
    def loss_info_fn(self, x, logits, zk, log_jac_det, mu, sd):
        # Note that the constant term -0.5*D*log(2*pi) is ignored because 
        # it will be canceled out when computing the regularized KL term
        Elogq0_z0_sum = - (sd.log() + 0.5).sum()          # (B,D) -> scalar
        Elogqk_zk_sum = Elogq0_z0_sum - log_jac_det.sum() # (B,)  -> scalar
        logp_zk_sum   = - 0.5 * (zk**2).sum()             # (B,D) -> scalar
        kld = (Elogqk_zk_sum - logp_zk_sum) / x.size(0)
        
        neg_logL = F.binary_cross_entropy_with_logits(    # (B,1,28,28)
            logits, x, reduction='sum') / x.size(0)       #       -> scalar
        return {'recon_err':neg_logL, 'regul_kld':kld}
    
    def get_anneal_beta(self):
        return min(1., self.anneal_init_beta + self.anneal_rate*self.iteration)
    
    def _train_loop(self, train_loader, optimizer):
        self.train()
        for x, _ in train_loader:
            self.iteration += 1
            x = x.to(self.device)
            logits, zk, log_jac_det, z0, mu, sd = self(x)
            loss_info = self.loss_info_fn(x, logits, zk, log_jac_det, mu, sd)
            if self.anneal_mode:
                beta = self.get_anneal_beta()
                if beta == 1.:
                    self.anneal_mode = False
                    print(f'Anneal mode is turned off at iteration {self.iteration}')
                loss = loss_info['recon_err'] + loss_info['regul_kld']*beta
            else:
                loss = loss_info['recon_err'] + loss_info['regul_kld']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    @torch.no_grad()
    def _validate_loop(self, valid_loader, seed_valid):
        self.eval()
        kld, neg_logL = [], []
        self.active_rng = torch.Generator(device=self.device).manual_seed(seed_valid)
        for x, _ in valid_loader:
            x = x.to(self.device)
            logits, zk, log_jac_det, z0, mu, sd = self(x)
            loss_info = self.loss_info_fn(x, logits, zk, log_jac_det, mu, sd)
            kld.append(loss_info['regul_kld'])
            neg_logL.append(loss_info['recon_err'])
        self.active_rng = self.rng_train
        kld_mean      = torch.stack(kld).mean()
        neg_logL_mean = torch.stack(neg_logL).mean()
        valid_loss = torch.stack((
            neg_logL_mean+kld_mean, neg_logL_mean, kld_mean))
        return valid_loss   # (3,)
    
    def train_and_validate(self, 
                           train_loader, valid_loader, 
                           num_epochs, num_epochs_anneal, num_epochs_adaptive_lr, 
                           anneal, anneal_init_beta, 
                           lr_init, lr_factor, lr_patience, 
                           seed_valid=27182818, 
                           log_every=1):
        t_init = time()
        
        # define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_init)
        
        # define scheduler - use ReduceLROnPlateau to control lr adaptively
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=lr_factor, 
            patience=lr_patience-1,    # offset by 1 due to different def of patience
            threshold=0., 
            threshold_mode='abs'
        )
        lr_current = scheduler.get_last_lr()
        
        if anneal:
            num_iters_per_epoch = len(train_loader)
            self.num_iters_anneal = num_epochs_anneal * num_iters_per_epoch
            self.anneal_rate = (1-anneal_init_beta) / self.num_iters_anneal
            self.anneal_init_beta = anneal_init_beta
            self.anneal_mode = True
            print(f'Anneal mode is turned on')
        
        # initial conditions
        valid_loss = self._validate_loop(valid_loader, seed_valid=seed_valid)
        valid_loss_lst = []
        valid_loss_lst.append(valid_loss)
        best_epoch = 0
        best_valid_loss = torch.nan_to_num(valid_loss[0], nan=torch.inf).item()
        best_valid_loss_lst = []
        best_valid_loss_lst.append(best_valid_loss)
        best_model_state_dict = deepcopy(self.state_dict())
        
        # start training
        num_plateaus = 0
        num_bad_epochs = 0
        max_num_bad_epochs = 0
        num_plateaus_fixed_lr = 0
        plateau_index = torch.full([num_epochs+1], -666, dtype=torch.int)
        plateau_info_lst = []
        plateau_info_lst.append(('x', num_plateaus, num_bad_epochs))  # initial state
        def format_plateau_info(plateau_info):     # e.g. v ( 0,  0)
            return f"{plateau_info[0]} ({plateau_info[1]:2d}, {plateau_info[2]:2d})"
        def format_best_info(epoch, best_loss):    # e.g. (  12,  46.718)
            return f"({epoch:4d}, {best_loss:7.3f})"
        def format_valid_loss_info(valid_loss):    # e.g. 632.262 (631.165,   1.090)
            return f"{valid_loss[0]:7.3f} ({valid_loss[1]:7.3f}, {valid_loss[2]:7.3f})"
        
        num_epochs_fixed_lr = num_epochs - num_epochs_adaptive_lr
        print(f"\nFor the first {num_epochs_fixed_lr} epochs, "
              f"the model is trained with a fixed learning rate.\n")

        # Column widths
        epoch_width = 8
        plateau_width = 13
        best_width = 18
        loss_triplet_width = 29
        time_width = 7
        if log_every > 0:
            # print header for columnar format
            print('Plateau         = status (#plateaus, #bad_epochs)')
            print('Best Validation = (best_epoch, best_valid_loss)\n')
            print(f"{'Epoch':<{epoch_width}}"
                  f"{'Plateau':<{plateau_width}}"
                  f"{'Best Validation':<{best_width}}"
                  f"{'ValidLoss (Recon, RegdKLD)':<{loss_triplet_width}}"
                  f"{'Time':<{time_width}}")
            # print initial validation loss
            runtime_str = format_runtime(time()-t_init, digits=0)
            plateau_str = format_plateau_info(plateau_info_lst[0])
            best_str = format_best_info(best_epoch, best_valid_loss)
            valid_loss_str = format_valid_loss_info(valid_loss)
            print(f"{'Initial':<{epoch_width}}"
                  f"{plateau_str:<{plateau_width}}"
                  f"{best_str:<{best_width}}"
                  f"{valid_loss_str:<{loss_triplet_width}}"
                  f"{runtime_str:<{time_width}}")
            t_log = time()
        
        for i_epoch in range(1, num_epochs+1):
            self._train_loop(train_loader, optimizer)
            valid_loss = self._validate_loop(valid_loader, seed_valid=seed_valid)
            valid_loss_lst.append(valid_loss)

            # check whether validation loss has improved
            if valid_loss[0].item() < best_valid_loss:
                # improved -> update best info
                best_epoch = i_epoch
                best_valid_loss = valid_loss[0].item()
                best_model_state_dict = deepcopy(self.state_dict())

                # improved -> before resetting, check whether the loss has plateaued
                if num_bad_epochs >= lr_patience:
                    # plateaued -> increase num_plateaus, update max_num_bad_epochs & plateau index
                    num_plateaus += 1
                    max_num_bad_epochs = max(max_num_bad_epochs, num_bad_epochs)
                    # plateaued then improved -> current epoch is NOT included
                    plateau_index[i_epoch-num_bad_epochs:i_epoch] = num_plateaus
                # improved -> reset counting
                num_bad_epochs = 0
                # improved -> create plateau_info tuple
                plateau_info = ('v', num_plateaus, num_bad_epochs)
            else:
                # not improved -> increase num_bad_epochs
                num_bad_epochs += 1
                # not improved -> create plateau_info tuple
                plateau_info = ('-', num_plateaus, num_bad_epochs)
            
            # record the best loss up to i_epoch & plateau_info
            best_valid_loss_lst.append(best_valid_loss)
            plateau_info_lst.append(plateau_info)
            
            # print validation loss
            if log_every > 0 and (i_epoch % log_every == 0):
                runtime_str = format_runtime(time() - t_log, digits=0)
                plateau_str = format_plateau_info(plateau_info)
                best_str = format_best_info(best_epoch, best_valid_loss)
                valid_loss_str = format_valid_loss_info(valid_loss)
                print(f"{i_epoch:<{epoch_width}}"
                      f"{plateau_str:<{plateau_width}}"
                      f"{best_str:<{best_width}}"
                      f"{valid_loss_str:<{loss_triplet_width}}"
                      f"{runtime_str:<{time_width}}")
                t_log = time()

            # use ReduceLROnPlateau to control lr adaptively
            if i_epoch == num_epochs_fixed_lr:
                # set the baseline for the scheduler
                scheduler.step(best_valid_loss)
                lr_current = scheduler.get_last_lr()
                # baseline setting -> before resetting, check whether the loss has plateaued
                if num_bad_epochs >= lr_patience:
                    # plateaued -> increase num_plateaus, update max_num_bad_epochs & plateau index
                    num_plateaus += 1
                    max_num_bad_epochs = max(max_num_bad_epochs, num_bad_epochs)
                    # plateaued and not improved -> current epoch is included
                    plateau_index[i_epoch+1-num_bad_epochs:i_epoch+1] = num_plateaus
                # baseline setting -> record num_plateaus in the fixed lr phase
                num_plateaus_fixed_lr = num_plateaus
                # baseline setting -> reset counting
                num_plateaus = 0
                num_bad_epochs = 0
                print(f"\nFrom now on, the learning rate will be reduced "
                      f"if the validation loss plateaus (patience={lr_patience})")
            elif i_epoch > num_epochs_fixed_lr:
                # adapt learning rate if loss has plateaued
                scheduler.step(valid_loss[0].item())

                # check whether the loss has plateaued, i.e., whether lr has been updated
                if scheduler.get_last_lr() != lr_current:
                    # plateaued -> increase num_plateaus, update max_num_bad_epochs & plateau index
                    num_plateaus += 1
                    max_num_bad_epochs = max(max_num_bad_epochs, num_bad_epochs)
                    # plateaued and not improved -> current epoch is included
                    plateau_index[i_epoch+1-num_bad_epochs:i_epoch+1] = num_plateaus
                    # plateaued -> reset counting
                    num_bad_epochs = 0
                    lr_current = scheduler.get_last_lr()
                    print(f'Plateau {num_plateaus} ----- '
                          f'At the end of Epoch {i_epoch}, '
                          f'the learning rate has reduced to {lr_current}\n')

        # save validation loss history as a pd.DataFrame
        df_valid_loss_hist = pd.DataFrame({
            'Epoch': range(num_epochs+1), 
            'plateau_info': plateau_info_lst, 
            'plateau_index': plateau_index, 
            'best_valid_loss': best_valid_loss_lst,  # best_valid_loss up to i_epoch
        })
        valid_loss_np = torch.stack(valid_loss_lst).cpu().numpy() # (num_epochs+1,3)
        df_valid_loss_hist[['valid_loss', 'recon_err', 'regul_kld']] = valid_loss_np
        
        # Training completed: switch back to training mode and print time usage
        self.train()
        runtime = time() - t_init
        runtime_str = format_runtime(runtime, digits=0)
        secs_per_epoch = runtime/num_epochs
        iters_per_sec = self.iteration/runtime
        print(f'Time for training {num_epochs} epochs ({self.iteration} iters):  {runtime_str}'
              f'  @  {secs_per_epoch:.2f} sec/epoch' 
              f'  @  {iters_per_sec:.2f} iter/sec')
        return {'best_model_state_dict': best_model_state_dict,
                'best_valid_loss': best_valid_loss,
                'best_epoch': best_epoch,
                'df_valid_loss_hist': df_valid_loss_hist,  # pd.DataFrame
                'runtime': runtime, 
                'runtime_str': runtime_str, 
                'secs_per_epoch': secs_per_epoch, 
                'iters_per_sec': iters_per_sec, 
                'num_plateaus_fixed_lr': num_plateaus_fixed_lr, 
                'num_plateaus_adaptive_lr': num_plateaus, 
                'max_num_bad_epochs':max_num_bad_epochs} 
    
    @torch.no_grad()
    def evaluate(self, test_data, IS_size=5000, seed_eval=27182818, reduction='mean', log_every=100):
        '''Evaluate model by importance sampling
            estimate marginal likelihood and ELBO on the test dataset
            then compute KL(q_K||posterior)
            report -ELBO, -logp(x), and KL(q_K||posterior)
        '''
        was_training = self.training
        self.eval()

        t_init = time()
        t_delta = time()
        test_loader = DataLoader(test_data, batch_size=1)
        neg_logpx_lst = []
        kld_vi_lst    = []
        neg_elbo_lst  = []
        self.active_rng = torch.Generator(device=self.device).manual_seed(seed_eval)
        # Print IS evaluation header
        if log_every > 0:
            print("\nIS evaluation:\n")
            print(f"{'Completed':>9}  {'Progress':>8}  {'Delta':>8}  {'Total':>8}")
        for i_test, (x, _) in enumerate(test_loader, start=1):
            x = x.to(self.device)
            mu, sd, context = self.encode(x)            # (1,D) <- (1,1,P,P)
            mu = mu.expand(IS_size, self.latent_dim)    # (IS,D)
            sd = sd.expand(IS_size, self.latent_dim)    # (IS,D)
            context = context.expand(IS_size, self.context_dim)
            zk, log_jac_det, z0 = self.draw_from_flow(mu, sd, context)
            logits = self.decode(zk)      # (IS,1,P,P) <- (IS,D)
            logL = - F.binary_cross_entropy_with_logits(
                logits, x.expand(logits.shape), 
                reduction='none').sum((-1,-2,-3))       # (IS,)
            # Compute log p(zk) under the prior N(0,I). 
            # Note that the constant term -0.5*D*log(2*pi) is ignored because 
            # it will be canceled out when computing log p(x, zk) - log q(zk|x)
            logp_zk  = - 0.5 * (zk**2).sum(-1)          # (IS,)
            quadra   = ((z0-mu)/sd)**2                  # (IS,D)
            logq0_z0 = -(sd.log() + 0.5*quadra).sum(-1) # (IS,)
            logqk_zk = logq0_z0 - log_jac_det           # (IS,)
            
            # marginal loglikelihood
            logpq = logL + logp_zk - logqk_zk           # (IS,)
            logpx = logpq.logsumexp(0) - math.log(IS_size)  # scalar
            neg_logpx_lst.append(-logpx)
            
            # negELBO, we could take expectation on one term
            #    E(quadra) = 1 (cf. logq0_z0 above)
            Elogq0_z0 = - (sd.log() + 0.5).sum(-1)  # (IS,)
            Elogqk_zk = Elogq0_z0 - log_jac_det     # (IS,)
            neg_elbo  = (Elogqk_zk - logp_zk - logL).mean()  # (IS,) -> scalar
            neg_elbo_lst.append(neg_elbo)
            
            # KL(q_K||posterior) = logp(x) - ELBO
            kld_vi_lst.append(logpx + neg_elbo)

            if torch.cuda.is_available() and i_test == 1:
                FlowVAE.print_cuda_memory()
            if log_every > 0 and i_test % log_every == 0:
                delta_time_str = format_runtime(time() - t_delta, digits=0)
                total_time_str = format_runtime(time() - t_init, digits=0)
                percent = int(i_test / len(test_loader) * 100)
                print(f"{i_test:9d}  {percent:5d}%    {delta_time_str:>8}  {total_time_str:>8}")
                t_delta = time()

        # Evaluation completed: switch back to previous mode and print time usage
        self.train(was_training)
        self.active_rng = self.rng_train
        runtime = time() - t_init
        runtime_str = format_runtime(runtime, digits=0)
        print(f'Time for estimating -ELBO, -logp(x), and KL(qK||posterior) '
              f'with {IS_size} IS draws per test image: {runtime_str}')
        result = torch.stack((torch.stack(neg_elbo_lst), 
                              torch.stack(neg_logpx_lst), 
                              torch.stack(kld_vi_lst)))    # (3, test_size)
        if reduction=='mean':
            metrics = result.mean(1)   # (3,)
        elif reduction=='none': 
            metrics = result           # (3, test_size)
        else:
            raise Exception('Invalid reduction')
        
        if torch.cuda.is_available():
            FlowVAE.print_cuda_memory()
        
        return {'metrics': metrics, 
                'runtime': runtime, 
                'runtime_str': runtime_str}
    
    @staticmethod
    def print_cuda_memory():
        torch.cuda.synchronize()
        GiB = 1024**3
        allocated = torch.cuda.memory_allocated() / GiB
        reserved = torch.cuda.memory_reserved() / GiB
        max_allocated = torch.cuda.max_memory_allocated() / GiB
        max_reserved = torch.cuda.max_memory_reserved() / GiB
        print("CUDA Memory Usage:\n")
        print(f"{'':12}  {'Current':>13}  {'Max':>13}")
        print(f"{'Allocated':>12}  {allocated:10.2f} GiB  {max_allocated:10.2f} GiB")
        print(f"{'Reserved':>12}  {reserved:10.2f} GiB  {max_reserved:10.2f} GiB\n")
    
    @torch.no_grad()
    def reconstruct(self, x, binary=True):
        """Reconstruct a given image, i.e., x_new ~ p(x|zk) where zk ~ q(zk|x_old)"""
        was_training = self.training
        self.eval()

        logits, zk, log_jac_det, z0, mu, sd = self(x)
        pi = F.sigmoid(logits)

        self.train(was_training)
        return (torch.round(pi), zk) if binary else (pi, zk)
    
    @torch.no_grad()
    def generate(self, size, binary=True, seed_gen=42):
        """generate a new image from the marginal, i.e., x_new ~ p(x|zk) where zk ~ prior N(0,I)"""
        was_training = self.training
        self.eval()

        self.active_rng = torch.Generator(device=self.device).manual_seed(seed_gen)
        zk = torch.randn(size, self.latent_dim, generator=self.active_rng, device=self.device)
        logits = self.decode(zk)
        pi = F.sigmoid(logits)

        self.train(was_training)
        self.active_rng = self.rng_train
        return (torch.round(pi), zk) if binary else (pi, zk)
    
    def loss_info_fn_(self, x, logits, zk, log_jac_det, mu, sd):
        Elogq0_z0_sum = - (sd.log() + 0.5).sum()          # (B,D) -> scalar
        Elogqk_zk_sum = Elogq0_z0_sum - log_jac_det.sum() # (B,)  -> scalar
        Elogqk_zk     = Elogqk_zk_sum / x.size(0)
        
        logL_sum = - F.binary_cross_entropy_with_logits(
            logits, x, reduction='sum')           # (B,1,28,28) -> scalar
        logp_zk_sum  = - 0.5 * (zk**2).sum()            # (B,D) -> scalar
        neg_logjoint = - (logL_sum + logp_zk_sum) / x.size(0)
        return {'Elogqk_zk':Elogqk_zk, 'neg_logjoint':neg_logjoint}

