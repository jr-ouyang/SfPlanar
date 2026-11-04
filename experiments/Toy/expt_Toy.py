import os
import torch
import pandas as pd
torch.set_default_dtype(torch.float64)

#------------------------------------------------------ parameter settings 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--output_path_csv', type=str)
parser.add_argument('--output_path_pt',  type=str)

parser.add_argument('--replicate',      type=int)
parser.add_argument('--num_layers_lst', type=int, default=[2,4,8,16,32], nargs='+')

parser.add_argument('--batch_size',           type=int, default=250)
parser.add_argument('--num_updates',          type=int, default=500_000)
parser.add_argument('--num_updates_per_loop', type=int, default=10_000)
parser.add_argument('--test_size',            type=int, default=1_000_000)
parser.add_argument('--batch_size_test',      type=int, default=100_000)

parser.add_argument('--lr_init',              type=float, default=0.001)
parser.add_argument('--lr_gamma',             type=float, default=0.95)

parser.add_argument('--seed_train', type=int, default=11235813)      # Fibonacci
parser.add_argument('--seed_test',  type=int, default=31415926)      # pi


args = parser.parse_args()
for arg in vars(args):
    print(f'{arg} = {getattr(args, arg)}')
print("\n")

print(f"dtype: {torch.get_default_dtype()}\n\n")

from sfplanar.vbayes import (
    PlanarFlow,
    FlowVBayes,
    TargetToy1,
    TargetToy2,
)

#------------------------------------------------------ simulation

checkpoint = {}
df_collection = []

for TargetToy in [TargetToy1, TargetToy2]:
    target_name = TargetToy.__name__[-4:]

    test_loss_collection = []
    for num_layers in args.num_layers_lst:
        for reparam in ['Old', 'New']:
            print(f"{target_name}    "
                  f"PlanarFlow({reparam}) with {num_layers} layers.\n")
            flow = PlanarFlow(dim=2, num_layers=num_layers, 
                              reparam=reparam, linear_layer=True, 
                              replicate=args.replicate)
            target = TargetToy()
            vbayes = FlowVBayes(flow=flow, target=target, seed_train=args.seed_train)
            
            results = vbayes.train_and_test(
                num_updates=args.num_updates, 
                num_updates_per_loop=args.num_updates_per_loop, 
                batch_size=args.batch_size,
                test_size=args.test_size,
                batch_size_test=args.batch_size_test,
                seed_test=args.seed_test,
                log_every=10,
            )

            # results = {'state_dict': ... , 
            #            'test_loss_hist': tensor of shape (num_loops+1,)}
            checkpoint[f"{target_name}-layers_{num_layers}-{reparam}"] = results

            test_loss_collection.append({
                "Replicate": args.replicate,
                "Toy": target_name[-1],
                "Num_Layers": num_layers,
                "Reparam": reparam,
                "KL": results['test_loss_hist'][-1].item(), 
            })

    df = pd.DataFrame(test_loss_collection)
    df_collection.append(df)

    # print
    print(f"{target_name} Results:")
    print(df, "\n\n")



for output_path in [args.output_path_csv, args.output_path_pt]:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)



df_combined = pd.concat(df_collection, axis=0, ignore_index=True)
df_combined.to_csv(args.output_path_csv, index=False)
torch.save(checkpoint, args.output_path_pt)

