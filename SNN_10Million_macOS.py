#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:47:57 2026

@author: rafalb
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 11:52:05 2026

@author: R. Lahoz-Beltra, The Crazy Lab©, rafa.lahoz@protonmail.ch
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Izhikevich Simulation - 10 MILLION synapses
Default Mode Network (DMN / "Self") and Qualia Proxy (Phi)
Optimized for Apple Silicon (MPS / Mac) or high-performance CPU
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 0. Device Selection (MPS GPU / CPU)
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.empty_cache()  # Limpiar memoria previa en Metal
    print("-> Computation: Apple Silicon GPU (MPS) ACTIVE")
else:
    device = torch.device("cpu")
    print("-> Computation: CPU (Note: Check PyTorch if you want to use GPU)")

# ==========================================
# 1. Network Configuration (10M Synapses)
# ==========================================
N_sensory = 5000  # Sensory Layer (0 to 4999)
N_dmn = 5000      # DMN / Self Layer (5000 to 9999)
N = N_sensory + N_dmn
K = 1000          # 1,000 connections per neuron -> 10,000 * 1,000 = 10,000,000 synapses

print(f"Total neurons (N): {N:,}")
print(f"Total synapses: {N * K:,}")

# Izhikevich parameters
re = torch.rand(N, device=device)
a = 0.02 * torch.ones(N, device=device)
b = 0.2 * torch.ones(N, device=device)
c = -65.0 + 15.0 * (re**2)
d = 8.0 - 6.0 * (re**2)

# Adjust autonomous pacemakers in the DMN (5000 to 6000)
c[5000:6000] = -55.0
d[5000:6000] = 4.0

print("Generating 10,000,000 synapse graph...")
t0 = time.time()

# Vectorized Connectivity Matrix (COO) - Optimización a int32 para ahorrar VRAM
rows = torch.randint(0, N, (N * K,), device=device, dtype=torch.int32)
cols = torch.repeat_interleave(torch.arange(N, device=device, dtype=torch.int32), K)

# Avoid self-connections
mask_self = rows == cols
rows[mask_self] = (rows[mask_self] + 1) % N

weights_factor = torch.where(cols >= 5000, 0.6, 0.4)
S_weights = weights_factor * torch.rand(N * K, device=device)
S_indices = torch.stack([rows.long(), cols.long()])  # Compatibilidad de índice para index_add_

print(f"Synapse graph built in {time.time() - t0:.2f} seconds.")

# State Variables
v = -65.0 * torch.ones(N, device=device)
u = b * v
STDP_trace = torch.zeros(N, device=device)

# Constant endogenous current in the DMN for the "Self"
I_tonic = torch.zeros(N, device=device)
I_tonic[5000:6000] = 7.0

# Spike registration
T = 900  # 300ms Rest | 300ms Stimulus | 300ms Rest
spike_matrix_gpu = torch.zeros((N, T), dtype=torch.bool, device=device)

# ==========================================
# 2. Subsampled Qualia Proxy (Phi)
# ==========================================
def calculate_qualia_proxy(firing_matrix_cpu, max_sampled=200, reg_factor=1e-2):
    """ Estimates Phi on a dynamic subset of the most active neurons """
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

# ==========================================
# 3. Main Simulation Loop
# ==========================================
print("Starting simulation of the 'Self' before, during, and after stimulus...")
t_sim_start = time.time()

for t in range(T):
    # External sensory stimulus (300ms to 600ms)
    I_ext = torch.zeros(N, device=device)
    if 300 <= t < 600:
        I_ext[0:1500] = 16.0  # Applied to 1500 sensory neurons

    I = I_tonic + I_ext + 3.0 * torch.randn(N, device=device)

    fired_mask = v >= 30.0
    fired_indices = torch.where(fired_mask)[0]
    
    if len(fired_indices) > 0:
        spike_matrix_gpu[fired_indices, t] = True
        v[fired_mask] = c[fired_mask]
        u[fired_mask] += d[fired_mask]
        
        # --- OPTIMIZACIÓN MPS / METAL ---
        # Máscara booleana para evitar asignaciones masivas de memoria con torch.isin
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

    # Izhikevich Integration
    v = v + 0.5 * (0.04 * v**2 + 5.0 * v + 140.0 - u + I + I_syn)
    v = v + 0.5 * (0.04 * v**2 + 5.0 * v + 140.0 - u + I + I_syn)
    u = u + a * (b * v - u)

print(f"Simulation calculated in {time.time() - t_sim_start:.2f} seconds.")

# ==========================================
# 4. Post-Processing and HD Plots
# ==========================================
print("Calculating $\Phi$ (Qualia) and generating final plots...")
history_spikes_cpu = spike_matrix_gpu.cpu().numpy().astype(np.uint8)

phi_history = np.zeros(T)
time_window = 80

for t in range(time_window, T):
    window_spikes = history_spikes_cpu[:, t - time_window : t]
    phi_history[t] = calculate_qualia_proxy(window_spikes)

# --- Visual Configuration ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

color_sensory = '#1f77b4'
color_dmn = '#d62728'
color_phi = '#8c564b'

for ax in (ax1, ax2, ax3):
    ax.axvspan(300, 600, color='#ffe6e6', alpha=0.5, label='_Stimulus')
    ax.axvline(300, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(600, color='green', linestyle='--', linewidth=1, alpha=0.7)

# Panel 1: Raster Plot (Subsampled 1 in every 5 neurons for clarity)
step_render = 5
sensory_spikes = history_spikes_cpu[:5000:step_render, :]
dmn_spikes = history_spikes_cpu[5000::step_render, :]

sy, sx = np.where(sensory_spikes > 0)
dy, dx = np.where(dmn_spikes > 0)

ax1.scatter(sx, sy * step_render, s=0.8, c=color_sensory, alpha=0.5, label='Sensory Layer (0-4999)')
ax1.scatter(dx, (dy * step_render) + 5000, s=0.8, c=color_dmn, alpha=0.5, label='DMN / Self Network (5000-9999)')
ax1.set_ylabel('Neuron ID', fontsize=10, fontweight='bold')
ax1.set_title('"Self" Spiking Neural Network (10,000 Neurons | 10M Synapses)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, N)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

# Panel 2: Average Firing Rate
firing_rate = np.mean(history_spikes_cpu, axis=0) * 1000
ax2.plot(np.arange(T), firing_rate, color='#2ca02c', linewidth=1.2)
ax2.set_ylabel('Activity (Hz)', fontsize=10, fontweight='bold')
ax2.set_title('Global Population Firing Rate', fontsize=10)
ax2.grid(True, alpha=0.3)

# Panel 3: Qualia Proxy (Phi)
ax3.plot(np.arange(T), phi_history, color=color_phi, linewidth=2.0, label='$\Phi$ (Qualia)')
ax3.fill_between(np.arange(T), phi_history, color=color_phi, alpha=0.2)
ax3.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
ax3.set_ylabel('$\Phi$ (bits)', fontsize=10, fontweight='bold')
ax3.set_title('Integrated Information ($\Phi$) - Consciousness Proxy', fontsize=10)
ax3.set_ylim(0, max(np.max(phi_history) * 1.15, 1.0))
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()