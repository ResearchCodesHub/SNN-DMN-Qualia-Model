#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:56:25 2026

@author: rafalb
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:48:01 2026

@author: R. Lahoz-Beltra, The Crazy Lab©, rafa.lahoz@protonmail.ch
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

# Hardware accelerator selection (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()
else:
    device = torch.device("cpu")

print(f"Executing simulation using device: {device}")

def run_single_simulation(seed=42, I_tonic_val=7.0, T=900, K=1000, N_sensory=5000, N_dmn=5000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = N_sensory + N_dmn
    
    # Izhikevich neuron parameters
    re = torch.rand(N, device=device)
    a = 0.02 * torch.ones(N, device=device)
    b = 0.2 * torch.ones(N, device=device)
    c = -65.0 + 15.0 * (re**2)
    d = 8.0 - 6.0 * (re**2)

    # DMN pacemaker neurons
    c[5000:6000] = -55.0
    d[5000:6000] = 4.0

    # Connectivity graph (10 million synapses) - Optimización con int32
    rows = torch.randint(0, N, (N * K,), device=device, dtype=torch.int32)
    cols = torch.repeat_interleave(torch.arange(N, device=device, dtype=torch.int32), K)

    mask_self = rows == cols
    rows[mask_self] = (rows[mask_self] + 1) % N

    weights_factor = torch.where(cols >= 5000, 0.6, 0.4)
    S_weights = weights_factor * torch.rand(N * K, device=device)
    S_indices = torch.stack([rows.long(), cols.long()])

    # Initial state variables
    v = -65.0 * torch.ones(N, device=device)
    u = b * v

    # Endogenous tonic current
    I_tonic = torch.zeros(N, device=device)
    I_tonic[5000:6000] = I_tonic_val

    spike_matrix_gpu = torch.zeros((N, T), dtype=torch.bool, device=device)

    for t in range(T):
        # External sensory stimulus (t in [300, 600] ms)
        I_ext = torch.zeros(N, device=device)
        if 300 <= t < 600:
            I_ext[0:1500] = 16.0

        I = I_tonic + I_ext + 3.0 * torch.randn(N, device=device)

        fired_mask = v >= 30.0
        fired_indices = torch.where(fired_mask)[0]
        
        if len(fired_indices) > 0:
            spike_matrix_gpu[fired_indices, t] = True
            v[fired_mask] = c[fired_mask]
            u[fired_mask] += d[fired_mask]
            
            # --- OPTIMIZACIÓN MPS / METAL (Reemplazo de torch.isin) ---
            spike_mask = torch.zeros(N, dtype=torch.bool, device=device)
            spike_mask[fired_indices] = True
            match_syn = spike_mask[S_indices[1]]

            if torch.any(match_syn):
                I_syn = torch.zeros(N, device=device)
                I_syn.index_add_(0, S_indices[0][match_syn], S_weights[match_syn])
            else:
                I_syn = 0.0
        else:
            I_syn = 0.0

        v = v + 0.5 * (0.04 * v**2 + 5.0 * v + 140.0 - u + I + I_syn)
        v = v + 0.5 * (0.04 * v**2 + 5.0 * v + 140.0 - u + I + I_syn)
        u = u + a * (b * v - u)

    result = spike_matrix_gpu.cpu().numpy().astype(np.uint8)
    
    # Limpieza de memoria de GPU tras cada simulación
    if device.type == "mps":
        torch.mps.empty_cache()
        
    return result

def calculate_qualia_proxy(firing_matrix_cpu, max_sampled=200, reg_factor=1e-2):
    std_per_neuron = np.std(firing_matrix_cpu, axis=1)
    active_indices = np.where(std_per_neuron > 1e-5)[0]
    
    if len(active_indices) < 10:
        return 0.0

    if len(active_indices) > max_sampled:
        top = np.argsort(std_per_neuron[active_indices])[-max_sampled:]
        active_indices = active_indices[top]

    sub_matrix = firing_matrix_cpu[active_indices, :]
    n_active = sub_matrix.shape[0]

    cov_full = np.cov(sub_matrix) + reg_factor * np.eye(n_active)
    mid = n_active // 2
    
    cov_A = np.cov(sub_matrix[:mid, :]) + reg_factor * np.eye(mid)
    cov_B = np.cov(sub_matrix[mid:, :]) + reg_factor * np.eye(n_active - mid)

    try:
        _, logdet_full = np.linalg.slogdet(cov_full)
        _, logdet_A = np.linalg.slogdet(cov_A)
        _, logdet_B = np.linalg.slogdet(cov_B)
        log_phi = logdet_A + logdet_B - logdet_full
        return max(0.0, float(0.5 * log_phi))
    except Exception:
        return 0.0

if __name__ == "__main__":
    N_SEEDS = 30
    T = 900
    time_window = 80

    print(f"=== Starting Multi-Seed Simulation Run (N = {N_SEEDS} Seeds) ===")
    start_total_time = time.time()

    phi_baseline_seeds = np.zeros((N_SEEDS, T))
    phi_ablation_seeds = np.zeros((N_SEEDS, T))

    for s in range(N_SEEDS):
        t0 = time.time()
        print(f"-> Processing Seed {s+1}/{N_SEEDS}...")
        
        # 1. Control Condition (I_tonic = 7.0 pA)
        spikes_ctrl = run_single_simulation(seed=s, I_tonic_val=7.0, T=T)
        for t in range(time_window, T):
            phi_baseline_seeds[s, t] = calculate_qualia_proxy(spikes_ctrl[:, t - time_window : t])
            
        # 2. Ablation Condition (I_tonic = 0.0 pA)
        spikes_ablation = run_single_simulation(seed=s, I_tonic_val=0.0, T=T)
        for t in range(time_window, T):
            phi_ablation_seeds[s, t] = calculate_qualia_proxy(spikes_ablation[:, t - time_window : t])
            
        print(f"   Completed Seed {s+1} in {time.time() - t0:.2f} seconds.")

    print(f"\nAll simulations completed in {(time.time() - start_total_time)/60:.2f} minutes.")

    # Statistical Aggregation
    mean_phi_ctrl = np.mean(phi_baseline_seeds, axis=0)
    std_phi_ctrl = np.std(phi_baseline_seeds, axis=0)

    mean_phi_abl = np.mean(phi_ablation_seeds, axis=0)
    std_phi_abl = np.std(phi_ablation_seeds, axis=0)

    # Statistical Hypothesis Testing (Paired t-tests)
    window_baseline = slice(0, 300)
    window_onset    = slice(300, 450)
    window_post     = slice(600, 900)

    ctrl_base = np.mean(phi_baseline_seeds[:, window_baseline], axis=1)
    abl_base  = np.mean(phi_ablation_seeds[:, window_baseline], axis=1)

    ctrl_onset = np.mean(phi_baseline_seeds[:, window_onset], axis=1)
    abl_onset  = np.mean(phi_ablation_seeds[:, window_onset], axis=1)

    ctrl_post = np.mean(phi_baseline_seeds[:, window_post], axis=1)
    abl_post  = np.mean(phi_ablation_seeds[:, window_post], axis=1)

    t_base, p_base = stats.ttest_rel(ctrl_base, abl_base)
    t_onset, p_onset = stats.ttest_rel(ctrl_onset, abl_onset)
    t_post, p_post = stats.ttest_rel(ctrl_post, abl_post)

    print("\n" + "="*60)
    print(" STATISTICAL HYPOTHESIS TESTING RESULTS (Paired t-tests, N = 30)")
    print("="*60)
    print("1. Baseline Resting State (0-300 ms):")
    print(f"   Control Mean: {np.mean(ctrl_base):.2f} +/- {np.std(ctrl_base):.2f} bits")
    print(f"   Ablation Mean: {np.mean(abl_base):.2f} +/- {np.std(abl_base):.2f} bits")
    print(f"   t(29) = {t_base:.4f}, p-value = {p_base:.4e}")
    
    print("\n2. Stimulus Onset Latency (300-450 ms):")
    print(f"   Control Mean: {np.mean(ctrl_onset):.2f} +/- {np.std(ctrl_onset):.2f} bits")
    print(f"   Ablation Mean: {np.mean(abl_onset):.2f} +/- {np.std(abl_onset):.2f} bits")
    print(f"   t(29) = {t_onset:.4f}, p-value = {p_onset:.4e}")
    
    print("\n3. Post-Stimulus Recovery (600-900 ms):")
    print(f"   Control Mean: {np.mean(ctrl_post):.2f} +/- {np.std(ctrl_post):.2f} bits")
    print(f"   Ablation Mean: {np.mean(abl_post):.2f} +/- {np.std(abl_post):.2f} bits")
    print(f"   t(29) = {t_post:.4f}, p-value = {p_post:.4e}")
    print("="*60 + "\n")

    # Results Plotting
    plt.figure(figsize=(11, 5.5), dpi=300)
    t_axis = np.arange(T)

    plt.plot(t_axis, mean_phi_ctrl, label="Control (I_tonic = 7.0 pA)", color="navy", linewidth=2)
    plt.fill_between(t_axis, mean_phi_ctrl - std_phi_ctrl, mean_phi_ctrl + std_phi_ctrl, color="navy", alpha=0.2, label="Control +/- 1 SD")

    plt.plot(t_axis, mean_phi_abl, label="Ablation (I_tonic = 0.0 pA)", color="crimson", linestyle="--", linewidth=2)
    plt.fill_between(t_axis, mean_phi_abl - std_phi_abl, mean_phi_abl + std_phi_abl, color="crimson", alpha=0.2, label="Ablation +/- 1 SD")

    plt.axvspan(300, 600, color="gray", alpha=0.15, label="Sensory Stimulus")
    plt.title(f"Integrated Information Dynamics Across N={N_SEEDS} Random Seeds: Control vs. Ablation", fontsize=12, fontweight="bold")
    plt.xlabel("Time (ms)", fontsize=11)
    plt.ylabel("Integrated Information Phi (bits)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.xlim(0, T)
    plt.tight_layout()
    plt.savefig("Figure_Ablation_Statistical_Analysis.png", dpi=300)
    plt.show()