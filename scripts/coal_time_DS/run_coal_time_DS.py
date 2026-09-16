# run_coal_time_DS.py
import argparse
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cftp_func import *

parser = argparse.ArgumentParser()
parser.add_argument('--graph_type', type=str, required=True, choices=['ER', 'RR', 'CW', 'SK'])
parser.add_argument('--coup_type', type=str, default='ising', choices=['ising', 'gauss'])
parser.add_argument('--d', type=int, default=4)
parser.add_argument('--beta', type=float, required=True)
parser.add_argument('--n_runs', type=int, default=25)
parser.add_argument('--max_iter', type=int, default=2**22)
parser.add_argument('--n_configs', type=int, default=10)
parser.add_argument('--lazy', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--N_list', type=int, nargs='+', default=[100, 500, 1000, 2500, 5000, 10000])

if __name__ == '__main__':
    args = parser.parse_args()
    lazy = args.lazy == 'True'

    # define G_factory based on graph type
    if args.graph_type == 'ER':
        G_factory = lambda N: create_ER_graph(N, args.d, args.coup_type)
    elif args.graph_type == 'RR':
        G_factory = lambda N: create_RR_graph(N, args.d, args.coup_type)
    elif args.graph_type == 'CW':
        G_factory = lambda N: create_complete_graph(N, 'CW')
    elif args.graph_type == 'SK':
        G_factory = lambda N: create_complete_graph(N, 'SK')

    df = compute_coal_time_DS(
        N_list=args.N_list,
        beta=args.beta,
        G_factory=G_factory,
        n_runs=args.n_runs,
        max_iter=args.max_iter,
        n_configs=args.n_configs,
        lazy=lazy,
    )

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, '..', '..', 'results', 'tmp')
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir,
        f"coal_time_DS_{args.graph_type}_{args.coup_type}_d{args.d}_beta{args.beta:.3f}.csv")


    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"Saved to {csv_path}")