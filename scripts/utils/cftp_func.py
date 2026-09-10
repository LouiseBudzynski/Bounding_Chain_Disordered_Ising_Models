"""
This script implements all needed functions for the project "Bounding Chains for Disordered Ising Models". 
"""

from numba import njit
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import norm
from scipy.sparse import csr_matrix


## MODELS
def create_ER_graph(N, d, coupling_type='ising'):
    G = nx.erdos_renyi_graph(N, d/N)
    if coupling_type == 'ising':
        couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
    elif coupling_type == 'gauss':
        couplings = np.triu(np.random.normal(0,1, size=(N, N)), k=1)
    else:
        raise ValueError(f"coupling_type must be 'ising' or 'gauss', got '{coupling_type}'")
    couplings += couplings.T  # symmetric couplings
    couplings = couplings * nx.to_numpy_array(G)
    return G, couplings

def create_RR_graph(N, d, coupling_type='ising'):
    G = nx.random_regular_graph(d, N)
    if coupling_type == 'ising':
        couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
    elif coupling_type == 'gauss':
        couplings = np.triu(np.random.normal(0,1, size=(N, N)), k=1)
    else:
        raise ValueError(f"coupling_type must be 'ising' or 'gauss', got '{coupling_type}'")
    couplings += couplings.T  # symmetric couplings
    couplings = couplings * nx.to_numpy_array(G)
    return G, couplings

def create_lattice_graph(L, d, model='RL_sg', periodic=True):
    if d not in [2, 3]:
        raise ValueError(f"d must be 2 or 3, got {d}")
    G = nx.grid_graph(dim=[L]*d, periodic=periodic)
    N = L**d
    if model=='RL_ferr':
        couplings = np.ones((N, N)) - np.eye(N)
    elif model=='RL_sg':
        couplings = np.triu(np.random.choice([-1, 1], size=(N, N)), k=1)
        couplings += couplings.T
        couplings = couplings * nx.to_numpy_array(G)
    else:
        raise ValueError(f"model must be 'RL_ferr' or 'RL_sg', got '{model}'")
    return G, couplings

def create_complete_graph(N, model='SK'):
    G = nx.complete_graph(N)
    if model=='CW':
        couplings = (np.ones((N, N)) - np.eye(N))/N
    elif model=='SK':
        couplings = np.random.normal(0, 1/np.sqrt(N), size=(N, N))
        couplings = (couplings + couplings.T) / 2
        np.fill_diagonal(couplings, 0)
    else:
        raise ValueError(f"model must be 'CW' or 'SK', got '{model}'")
    return G, couplings

## DYNAMICS
@njit
def F_beta_Glauber(beta, U):
    return 1 / (1 + np.exp(beta * U))

@njit
def F_beta_Metropolis(beta, U):
    if U <= 0:
        return 1
    else:
        return np.exp(-beta * U)

## Build CSR format
def _build_csr_slow(G, couplings):
    N = G.number_of_nodes()
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    edges_count = sum(1 for v in node_list for _ in G.neighbors(v))
    
    adj_indices = np.zeros(edges_count, dtype=np.int32)
    adj_weights = np.zeros(edges_count, dtype=np.float64)
    adj_abs_weights = np.zeros(edges_count, dtype=np.float64)
    adj_indptr = np.zeros(N + 1, dtype=np.int32)
    
    ptr = 0
    for i, v_node in enumerate(node_list):
        adj_indptr[i] = ptr
        for u_node in G.neighbors(v_node):
            j = node_to_idx[u_node]
            w = couplings[j, i]
            aw = np.abs(w)
            adj_indices[ptr] = j
            adj_weights[ptr] = w
            adj_abs_weights[ptr] = aw
            ptr += 1
    adj_indptr[N] = ptr
    
    return N, adj_indices, adj_indptr, adj_weights, adj_abs_weights    

def _build_csr(G, couplings): ### TOCHECK
    N = G.number_of_nodes()
    node_list = list(G.nodes())
    
    A = nx.to_scipy_sparse_array(G, nodelist=node_list, format='csr')
    
    adj_indptr = A.indptr.astype(np.int32)
    adj_indices = A.indices.astype(np.int32)
    
    # fill weights using the coupling matrix
    coup = nx.to_numpy_array(G, nodelist=node_list) * couplings
    adj_weights = coup[A.indices, np.repeat(np.arange(N), np.diff(A.indptr))].astype(np.float64)
    adj_abs_weights = np.abs(adj_weights)
    
    return N, adj_indices, adj_indptr, adj_weights, adj_abs_weights

def test_build_csr(G, couplings):
    # run both
    N_slow, idx_slow, indptr_slow, w_slow, aw_slow = _build_csr_slow(G, couplings)
    N_fast, idx_fast, indptr_fast, w_fast, aw_fast = _build_csr(G, couplings)

    # compare
    print(np.allclose(idx_slow, idx_fast))
    print(np.allclose(indptr_slow, indptr_fast))
    print(np.allclose(w_slow, w_fast))
    print(np.allclose(aw_slow, aw_fast))

## Forward BC
@njit
def _BC_fwd_numba_core(beta, N, adj_indices, adj_indptr, adj_weights, adj_abs_weights, max_iter, init_flag):
    config = np.empty(N, dtype=np.int8)
    
    # 1. Initialize Configuration based on flag
    if init_flag == 0:
        # FULL STAR: All nodes start as 0
        for i in range(N):
            config[i] = 0
        num_stars = N
    else:
        # SINGLE STAR: One node is 0, rest are random +/- 1
        for i in range(N):
            config[i] = 1 if np.random.rand() < 0.5 else -1
        star_node = np.random.randint(0, N)
        config[star_node] = 0
        num_stars = 1
        
    H_bar = np.zeros(N, dtype=np.float64)
    M = np.zeros(N, dtype=np.float64)
    
    # 2. Universal Field Initialization (handles ANY starting config)
    for v in range(N):
        state_v = config[v]
        start_idx = adj_indptr[v]
        end_idx = adj_indptr[v + 1]
        
        for i in range(start_idx, end_idx):
            u = adj_indices[i]
            w = adj_weights[i]
            aw = adj_abs_weights[i]
            
            # Inject v's state into its neighbor u
            if state_v != 0:
                H_bar[u] += w * state_v
            else:
                M[u] += aw

    nb_star_state = np.zeros(max_iter + 1, dtype=np.int32)
    nb_star_state[0] = num_stars
    
    # 3. Main MCMC Loop
    for timestep in range(max_iter):
        v = np.random.randint(0, N)
        r = np.random.rand()
        
        current_state = config[v]

        h_bar = H_bar[v]
        m = M[v]
        h_min = h_bar - m
        h_max = h_bar + m

        # Evaluate Transition
        if r < F_beta_Glauber(beta, -2 * h_min):
            new_state = 1
        elif r > F_beta_Glauber(beta, -2 * h_max):
            new_state = -1
        else:
            new_state = 0

        # Apply Updates
        if current_state != new_state:
            config[v] = new_state

            if current_state == 0:
                num_stars -= 1
            elif new_state == 0:
                num_stars += 1

            # Loop through neighbors
            start_idx = adj_indptr[v]
            end_idx = adj_indptr[v + 1]
            
            for i in range(start_idx, end_idx):
                u = adj_indices[i]
                w = adj_weights[i]
                aw = adj_abs_weights[i]

                # Strip old state
                if current_state != 0:
                    H_bar[u] -= w * current_state
                else:
                    M[u] -= aw

                # Inject new state
                if new_state != 0:
                    H_bar[u] += w * new_state
                else:
                    M[u] += aw

        nb_star_state[timestep + 1] = num_stars

        # Coalescence check
        if num_stars == 0:
            return nb_star_state[:timestep + 2], float(timestep)

    return nb_star_state, np.nan

@njit
def _BC_fwd_lazy_numba_core(beta, N, adj_indices, adj_indptr, adj_weights, adj_abs_weights, max_iter, init_flag, sampler):
    config = np.empty(N, dtype=np.int8)
    
    # 1. Initialize Configuration based on flag
    if init_flag == 0:
        # FULL STAR: All nodes start as 0
        for i in range(N):
            config[i] = 0
        num_stars = N
    else:
        # SINGLE STAR: One node is 0, rest are random +/- 1
        for i in range(N):
            config[i] = 1 if np.random.rand() < 0.5 else -1
        star_node = np.random.randint(0, N)
        config[star_node] = 0
        num_stars = 1
        
    H_bar = np.zeros(N, dtype=np.float64)
    M = np.zeros(N, dtype=np.float64)
    
    # 2. Universal Field Initialization (handles ANY starting config)
    for v in range(N):
        state_v = config[v]
        start_idx = adj_indptr[v]
        end_idx = adj_indptr[v + 1]
        
        for i in range(start_idx, end_idx):
            u = adj_indices[i]
            w = adj_weights[i]
            aw = adj_abs_weights[i]
            
            # Inject v's state into its neighbor u
            if state_v != 0:
                H_bar[u] += w * state_v
            else:
                M[u] += aw

    nb_star_state = np.zeros(max_iter + 1, dtype=np.int32)
    nb_star_state[0] = num_stars
    
    # 3. Main MCMC Loop
    for timestep in range(max_iter):
        v = np.random.randint(0, N)
        s = 1 if np.random.rand() < 0.5 else -1
        r = np.random.rand()
        
        current_state = config[v]

        if current_state != s:
            h_bar = H_bar[v]
            m = M[v]

            h_plus = h_bar + s * m
            h_minus = h_bar - s * m
    
            # Evaluate Transition
            if r < sampler(beta, -2 * s * h_minus):
                new_state = s
            elif r > sampler(beta, -2 * s * h_plus) and current_state == -s:
                new_state = -s
            else:
                new_state = 0

            # Apply Updates
            if current_state != new_state:
                config[v] = new_state

                if current_state == 0:
                    num_stars -= 1
                elif new_state == 0:
                    num_stars += 1

                # Loop through neighbors
                start_idx = adj_indptr[v]
                end_idx = adj_indptr[v + 1]
                
                for i in range(start_idx, end_idx):
                    u = adj_indices[i]
                    w = adj_weights[i]
                    aw = adj_abs_weights[i]

                    # Strip old state
                    if current_state != 0:
                        H_bar[u] -= w * current_state
                    else:
                        M[u] -= aw

                    # Inject new state
                    if new_state != 0:
                        H_bar[u] += w * new_state
                    else:
                        M[u] += aw

        nb_star_state[timestep + 1] = num_stars

        # Coalescence check
        if num_stars == 0:
            return nb_star_state[:timestep + 2], float(timestep)

    return nb_star_state, np.nan

def BC_fwd(beta, G, couplings, sampler=F_beta_Glauber, max_iter=100, init_mode='full_star', lazy=True):
    # Check: Metropolois sampler only works with lazy BC
    if not lazy and sampler is F_beta_Metropolis:
        raise ValueError(f"sampler '{sampler.__name__}' incompatible with lazy={lazy} BC update")    

    """
    init_mode options:
      'full_star'   : All nodes initialize to the star state (0)
      'single_star' : One node initialize to star state, the rest initialize to +/- 1
    """
    # Translate string mode to integer flag for Numba
    if init_mode == 'full_star':
        init_flag = 0
    elif init_mode == 'single_star':
        init_flag = 1
    else:
        raise ValueError(f"init_mode must be either 'full_star' or 'single_star', got '{init_mode}'")

    # build CSR format
    N, adj_indices, adj_indptr, adj_weights, adj_abs_weights = _build_csr(G, couplings)
    
    # Pass everything to the core
    if lazy:
        nb_star_state, final_step = _BC_fwd_lazy_numba_core(beta, N, adj_indices, adj_indptr, adj_weights, adj_abs_weights, max_iter, init_flag, sampler)
    else:
        nb_star_state, final_step = _BC_fwd_numba_core(beta, N, adj_indices, adj_indptr, adj_weights, adj_abs_weights, max_iter, init_flag)

        
    if not np.isnan(final_step):
        print(f"Coalescence achieved at iteration {int(final_step)} using {init_mode} initialization.")
    else:
        print(f"Warning: Coalescence not achieved after {max_iter} iterations.")
        
    return list(nb_star_state), final_step

## COALESCENCE TIME
def compute_coal_time(N_list, beta, G_factory, n_runs=5, sampler=F_beta_Glauber, max_iter=100, init_mode='full_star', lazy=False):
    results = []
    for N in N_list:
        print(f"N={int(N)}")
        coal_times = []
        for _ in range(n_runs):
            G, couplings = G_factory(N)  # resample each run for disorder average
            _, coal_time = BC_fwd(beta, G, couplings, sampler=sampler, max_iter=max_iter, init_mode=init_mode, lazy=lazy)
            if not np.isnan(coal_time):
                coal_times.append(coal_time)
        results.append({
            'beta': beta,
            'sampler': sampler.__name__,
            'lazy': lazy,
            'init_mode': init_mode,
            'n_runs': n_runs,
            'max_iter': max_iter,
            'N': N,
            'n_coal': len(coal_times),
            'mean_coal_time': np.mean(coal_times) if len(coal_times) > 0 else np.nan,
            'std_coal_time': np.std(coal_times) if len(coal_times) > 0 else np.nan,
        })
    return pd.DataFrame(results)

## ANALYSIS
def wilson_interval(k, n, confidence=0.95): ### TOCHECK
    z = norm.ppf((1 + confidence) / 2)
    p = k / n
    center = (k + z**2/2) / (n + z**2)
    margin = z * np.sqrt(k*(n-k)/n + z**2/4) / (n + z**2)
    lower = center - margin
    upper = center + margin
    return p, p - lower, upper - p  # value, err_down, err_up