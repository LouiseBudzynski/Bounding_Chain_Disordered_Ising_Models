Run run_nb_star_DS.py as:

bash
nohup python run_nb_star_DS.py --graph_type ER --coup_type gauss --d 4 --beta_list 0.3 0.31 0.32 0.33 0.34 --n_runs 50 --max_iter 100000000 --lazy False --N 1000 --n_points 1000 &

To launch multiple betas in parallel:

bash
for N in 100 1000 10000; do
    python run_nb_star_DS.py --graph_type ER --coup_type ising --d 4 --beta_list 0.2 0.21 0.22 --n_runs 25 --n_configs 10 --lazy False --N $N &
done

And to load all results in the notebook:

python
import glob
dfs = [pd.read_csv(f) for f in glob.glob('../results/tmp/nb_star_DS_ER_ising_d4_N*.csv')]
df = pd.concat(dfs, ignore_index=True)

Lauch slurm jobs:

sbatch slurm_nb_star_DS.batch ER ising 4 "0.2 0.3" 25 4194304 100 False 10000
