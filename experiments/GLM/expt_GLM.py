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
parser.add_argument('--num_layers_lst', type=int, default=[10,20,30,40], nargs='+')
parser.add_argument('--n_data',         type=int, default=[10,20], nargs=2)
parser.add_argument('--beta_dim',       type=int, default=10)
parser.add_argument('--beta_nzz',       type=int, default=2)
parser.add_argument('--rho_X',          type=float, default=0.5)
parser.add_argument('--prior_scale',    type=float, default=0.1)

parser.add_argument('--batch_size',           type=int, default=250)
parser.add_argument('--num_updates',          type=int, default=500_000)
parser.add_argument('--num_updates_per_loop', type=int, default=10_000)
parser.add_argument('--test_size',            type=int, default=1_000_000)
parser.add_argument('--batch_size_test',      type=int, default=100_000)
parser.add_argument('--IS_size',              type=int, default=1_000_000)
parser.add_argument('--IS_num_batches',       type=int, default=10)
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
    TargetBayesianRegression,
    UniformCauchyPrior,
    LinearRegressionLikelihood,
    LogisticRegressionLikelihood,
    DGP,
    estimate_log_evidence,
)

#------------------------------------------------------ simulation

checkpoint = {}
df_collection = []

for glm in ['Linear', 'Logistic']:
    if glm == 'Linear':
        sigma = 1.0
        raw_data, _ = DGP().generate_data_linear(sample_size=args.n_data[0], 
                                                 beta_dim=args.beta_dim, 
                                                 beta_nnz=args.beta_nzz, 
                                                 sigma=sigma, 
                                                 rho=args.rho_X,
                                                 replicate=args.replicate)
        likelihood = LinearRegressionLikelihood(sigma=sigma)
    else:
        raw_data, _ = DGP().generate_data_logistic(sample_size=args.n_data[1], 
                                                   beta_dim=args.beta_dim, 
                                                   beta_nnz=args.beta_nzz, 
                                                   rho=args.rho_X,
                                                   replicate=args.replicate)
        likelihood = LogisticRegressionLikelihood()

    prior = UniformCauchyPrior(beta_dim=args.beta_dim, scale=0.1)

    test_loss_collection = []
    for num_layers in args.num_layers_lst:
        for reparam in ['Old', 'New']:
            print(f"{glm} regression:")
            print(f"        PlanarFlow({reparam}) with {num_layers} layers.\n")
            flow = PlanarFlow(dim=args.beta_dim, num_layers=num_layers, 
                              reparam=reparam, linear_layer=True, 
                              replicate=args.replicate)
            target = TargetBayesianRegression(raw_data=raw_data, 
                                              likelihood=likelihood,
                                              prior=prior)
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
            checkpoint[f"{glm}-layers_{num_layers}-{reparam}"] = results

            test_loss_collection.append({
                "replicate": args.replicate,
                "glm": glm,
                "num_layers": num_layers,
                "reparam": reparam,
                "test_loss": results['test_loss_hist'][-1].item(), 
            })

    df = pd.DataFrame(test_loss_collection)

    # find the best performing model (lowest test_loss)
    best_idx = df["test_loss"].idxmin()
    best_row = df.loc[best_idx]
    best_num_layers = best_row["num_layers"]
    best_reparam = best_row["reparam"]
    print(f"Best performing model:\n"
          f"        PlanarFlow({best_reparam}) with {best_num_layers} layers.\n\n")

    # use the best performing model to estimate the log_evidence
    best_checkpoint = checkpoint[f"{glm}-layers_{best_num_layers}-{best_reparam}"]
    best_state_dict = best_checkpoint['state_dict']
    flow = PlanarFlow(dim=args.beta_dim, num_layers=best_num_layers, 
                      reparam=best_reparam, linear_layer=True, 
                      replicate=args.replicate)
    target = TargetBayesianRegression(raw_data=raw_data, 
                                      likelihood=likelihood,
                                      prior=prior)
    vbayes = FlowVBayes(flow=flow, target=target, seed_train=args.seed_train)
    vbayes.load_state_dict(best_state_dict)
    est_results = estimate_log_evidence(vbayes.flow, vbayes.target, 
                                        IS_size=args.IS_size, 
                                        num_batches=args.IS_num_batches, 
                                        seed=args.seed_test)
    
    ELBO = est_results['ELBO'].item()
    print(f"Sanity Check: ELBO = -TestLoss?\n"
          f"    ELBO:     {ELBO:>12.7f}\n"
          f"    TestLoss: {best_row['test_loss']:>12.7f}\n\n")

    # save KL into the dataframe
    df["log_evidence"] = est_results['log_evidence'].item()
    df["KL"] = df["log_evidence"] + df['test_loss']
    df_collection.append(df)

    # print
    print(f"{glm} Regression Results:")
    print(df, "\n\n")



for output_path in [args.output_path_csv, args.output_path_pt]:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)



df_combined = pd.concat(df_collection, axis=0, ignore_index=True)
df_combined.to_csv(args.output_path_csv, index=False)
torch.save(checkpoint, args.output_path_pt)

