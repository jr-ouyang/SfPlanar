import os
import torch

#------------------------------------------------------ parameters by argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str)
parser.add_argument('--flow_type', type=str, choices=['New', 'Old', 'IAF', 'Syl', 'CNF'])

# common parameters
parser.add_argument('--latent_dim',   type=int, default=20)
parser.add_argument('--context_dim',  type=int, default=16)
parser.add_argument('--num_layers',   type=int)
parser.add_argument('--num_channels', type=int, default=[16,32], nargs='+')

# IAF parameters
parser.add_argument('--hidden_dim', type=int, default=200)

# Syl parameters
parser.add_argument('--num_ortho_vecs', type=int, default=12)

# CNF parameters
parser.add_argument('--ode_method', type=str,   default='dopri5')
parser.add_argument('--atol',       type=float, default=1e-7)
parser.add_argument('--rtol',       type=float, default=1e-7)

# training parameters
parser.add_argument('--batch_size',             type=int, default=250)
parser.add_argument('--num_epochs',             type=int, default=1250)
parser.add_argument('--num_epochs_anneal',      type=int, default=125)
parser.add_argument('--num_epochs_adaptive_lr', type=int, default=125)
parser.add_argument('--anneal_init_beta', type=float, default=0.01)
parser.add_argument('--lr_init',          type=float, default=0.001)
parser.add_argument('--lr_factor',        type=float, default=0.5)
parser.add_argument('--lr_patience',      type=int,   default=10)

# seeds
parser.add_argument('--seed_data',      type=int, default=42, help='for stratified split and train_loader')
parser.add_argument('--seed_flow_init', type=int, default=1113)     # concatenated 2-digit primes
parser.add_argument('--seed_vae_init',  type=int, default=11235813) # Fibonacci
parser.add_argument('--seed_train',     type=int, default=31415926) # Pi
parser.add_argument('--seed_valid',     type=int, default=27182818) # e


# gather all parameters
args = parser.parse_args()
for arg in vars(args):
    print(f'{arg} = {getattr(args, arg)}')
print()


# import flow module 
if args.flow_type in ['New', 'Old']:
    from sfplanar.vae import ContextualPlanarFlow
elif args.flow_type == 'CNF':
    from sfplanar.vae import ContextualCNF
elif args.flow_type == 'Syl':
    from sfplanar.vae import ContextualSylvesterFlow
elif args.flow_type == 'IAF':
    from sfplanar.vae import ContextualIAF
else:
    raise Exception("Invalid flow_type")

from sfplanar.vae import FlowVAE


#------------------------------------------------------ load data
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, Subset


class ToBinary:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.round(x)


train_data = datasets.MNIST(
    root = 'data',
    train = True, 
    download=True,
    transform=Compose([
        ToTensor(), 
        ToBinary()
    ])
)
valid_size = 10000  # valid_size = test_size = 10000 -> train_size = 50000


# get the stratified split index by sklearn
from sklearn.model_selection import train_test_split
targets = train_data.targets
train_idx, valid_idx= train_test_split(list(range(len(targets))), 
                                       test_size=valid_size, 
                                       random_state=args.seed_data, 
                                       shuffle=True, 
                                       stratify=targets)

# split dataset into training set and validation set 
train_set = Subset(train_data, train_idx)
valid_set = Subset(train_data, valid_idx)


#------------------------------------------------------ define and train model
# Check cuda.is_available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device {device} \n')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# define flow and compute the number of parameters
torch.manual_seed(args.seed_flow_init)
if args.flow_type in ['New', 'Old']:
    flow = ContextualPlanarFlow(args.latent_dim, args.context_dim, 
                                num_layers=args.num_layers, 
                                reparam=args.flow_type)
    flow_name_in_df = args.flow_type
    num_params = args.num_layers*(2*args.latent_dim+1)*(args.context_dim+1)
elif args.flow_type == 'CNF':
    flow = ContextualCNF(args.latent_dim, args.context_dim, 
                         num_layers=args.num_layers, 
                         ode_method=args.ode_method, atol=args.atol, rtol=args.rtol)
    flow_name_in_df = args.flow_type
    num_params = args.num_layers*(3*args.latent_dim+1)*(args.context_dim+1)
elif args.flow_type == 'Syl':
    flow = ContextualSylvesterFlow(args.latent_dim, args.context_dim, 
                                   num_ortho_vecs=args.num_ortho_vecs, 
                                   num_layers=args.num_layers) 
    flow_name_in_df = args.flow_type + f'-{args.num_ortho_vecs}'
    D, M = args.latent_dim, args.num_ortho_vecs
    num_params_per_layer = (M*D + M*(M+1) + M) * (args.context_dim + 1)
    num_params = args.num_layers*num_params_per_layer
elif args.flow_type == 'IAF':
    flow = ContextualIAF(args.latent_dim, args.context_dim, 
                         hidden_dim=args.hidden_dim, 
                         num_layers=args.num_layers, 
                         activation=torch.nn.ELU)
    flow_name_in_df = args.flow_type + f'-{args.hidden_dim}'
    D, H = args.latent_dim, args.hidden_dim
    num_params_per_made = 0.5*(H**2 + 3*H*D) + 2*H + 2*D
    num_params_made = num_params_per_made * args.num_layers
    num_params_amor = (args.context_dim + 1) * H
    num_params = num_params_made + num_params_amor
else:
    raise Exception("Invalid flow_type")


# define VAE with flow
torch.manual_seed(args.seed_vae_init)
model = FlowVAE(args.num_channels, flow, seed_train=args.seed_train).to(device)


# define dataloader
g_loader = torch.Generator().manual_seed(args.seed_data)
train_loader = DataLoader(train_set, batch_size=args.batch_size, 
                          shuffle=True, generator=g_loader)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size)


# train and validate
results = model.train_and_validate(
    train_loader, valid_loader, 
    args.num_epochs, args.num_epochs_anneal, args.num_epochs_adaptive_lr, 
    anneal=True, anneal_init_beta=args.anneal_init_beta, 
    lr_init=args.lr_init, lr_factor=args.lr_factor, lr_patience=args.lr_patience, 
    seed_valid=args.seed_valid,
    log_every=1
)

print(f"Best epoch: {results['best_epoch']} \n")


#------------------------------------------------------------ save checkpoint

checkpoint = {
    'best_model_state_dict': results['best_model_state_dict'],
    'best_valid_loss': results['best_valid_loss'], 
    'best_epoch': results['best_epoch'], 
    'df_valid_loss_hist': results['df_valid_loss_hist'],
    'runtime': results['runtime'],
    'runtime_str': results['runtime_str'],
    'secs_per_epoch': results['secs_per_epoch'], 
    'iters_per_sec': results['iters_per_sec'], 
    'num_plateaus_fixed_lr': results['num_plateaus_fixed_lr'], 
    'num_plateaus_adaptive_lr': results['num_plateaus_adaptive_lr'], 
    'max_num_bad_epochs': results['max_num_bad_epochs'], 
    
    'seed_flow_init': args.seed_flow_init,
    'seed_vae_init': args.seed_vae_init,
    'seed_train': args.seed_train,
    'seed_valid': args.seed_valid,
}

output_dir = os.path.dirname(args.output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

torch.save(checkpoint, args.output_path)

