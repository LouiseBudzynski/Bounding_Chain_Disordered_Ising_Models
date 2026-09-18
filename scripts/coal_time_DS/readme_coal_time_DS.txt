Run run_coal_time_DS.py as:

bash
nohup python run_coal_time_DS.py --graph_type ER --coup_type ising --d 4 --beta 0.3 --n_runs 25 --n_configs 10 --lazy False --N_list 100 500 1000 2500

To launch multiple betas in parallel:

bash
for beta in 0.1 0.2 0.3 0.4 0.5; do
    python run_coal_time_DS.py --graph_type ER --coup_type ising --d 4 --beta $beta --n_runs 25 --n_configs 10 --lazy False --N_list 100 500 1000 2500 &
done

And to load all results in the notebook:

python
import glob
dfs = [pd.read_csv(f) for f in glob.glob('../results/tmp/coal_time_DS_ER_ising_d4_beta*.csv')]
df = pd.concat(dfs, ignore_index=True)

Lauch slurm jobs:

sbatch slurm_coal_time_DS.batch ER ising 4 0.3 25 4194304 100 False "100 500 1000 2500 5000 10000"

and

sbatch --array=0-4 slurm_array_coal_time_DS.batch ER ising 4 "0.1,0.2,0.3,0.4,0.5" 25 4194304 100 False "100 500 1000 2500 5000 10000"
