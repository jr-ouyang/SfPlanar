import os
import torch
import pandas as pd

#------------------------------------------------------ parameters by argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
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

# evaluation parameters
parser.add_argument('--IS_size',    type=int,   default=100_000, 
                    help='importance sampling size for estimating likelihood')

# seeds
parser.add_argument('--seed_flow_init', type=int, default=1113)     # concatenated 2-digit primes
parser.add_argument('--seed_vae_init',  type=int, default=11235813) # Fibonacci
parser.add_argument('--seed_train',     type=int, default=31415926) # Pi
parser.add_argument('--seed_valid',     type=int, default=27182818) # e
parser.add_argument('--seed_eval',      type=int, default=27182818) # e


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


class ToBinary:
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.round(x)


test_data = datasets.MNIST(
    root = 'data',
    train = False, 
    download=True,
    transform=Compose([
        ToTensor(), 
        ToBinary()
    ])
)


#------------------------------------------------------ define and train models
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


# load the trained model
checkpoint = torch.load(args.input_path, weights_only=False, map_location="cpu")
model.load_state_dict(checkpoint['best_model_state_dict'])


# estimate -ELBO, -logp(x), and KL[ q(z|x) || p(z|x) ]
evaluate_results = model.evaluate(test_data, 
                                  IS_size=args.IS_size, 
                                  seed_eval=args.seed_eval, 
                                  reduction='mean', log_every=100)

print(f"Evaluation metrics are {evaluate_results['metrics']} \n")  # tensor (3,)


#------------------------------------------------------------ save model performance

df_performance = pd.DataFrame([{
    'Flow':       flow_name_in_df, 
    'num_layers': args.num_layers, 
    'num_params': num_params, 
    
    '-ELBO':       evaluate_results['metrics'][0].item(), 
    '-logp(x)':    evaluate_results['metrics'][1].item(), 
    'KLD' :        evaluate_results['metrics'][2].item(), 
    'runtime':     evaluate_results['runtime'], 
    'runtime_str': evaluate_results['runtime_str'],
    
    'seed_flow_init': checkpoint['seed_flow_init'],
    'seed_vae_init':  checkpoint['seed_vae_init'],
    'seed_train':     checkpoint['seed_train'],
    'seed_valid':     checkpoint['seed_valid'],
    'seed_eval':      args.seed_eval,
}])


output_dir = os.path.dirname(args.output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

df_performance.to_csv(args.output_path, index=False)

