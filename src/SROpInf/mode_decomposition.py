"""mode_decomposition.py: Decompose a trajectory into modes, e.g., via POD."""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from typing import Tuple

from SROpInf.typing import Matrix

def pod(Q_scaled: Matrix,
        r_tentative: int,
        r: int = None, # if None, then use the cumulative energy ratio criterion to determine r; otherwise, use the user-input
        cumulative_energy_ratio: float = 0.996) -> Tuple[Matrix, int]:
    """Perform POD on the trajectory-wise-stacked given snapshot matrix Q (N, T*M) and return the leading modes.
    
    N: state dimension;
    T: number of time snapshots per trajectory;
    M: number of trajectories.
    
    Q needs to be pre-scaled to be Q_scaled, such that the grid-based norm of Q = the standard Euclidean norm of Q_scaled
    """
    
    print(f"Performing POD with tentative r={r_tentative}...")
    
    U, S, _ = sp.linalg.svd(Q_scaled, full_matrices=False)
    cumulative_energy_proportion = 100 * np.cumsum(S ** 2) / np.sum(S ** 2)
    
    # Plot the singular value decay and cumulative energy proportion
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(S[:r_tentative], 'o-')
    plt.xlabel('Mode Index')
    plt.ylabel('Singular Value')
    plt.title('Singular Value Decay')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(cumulative_energy_proportion, 'o-')
    plt.xlabel('Mode Index')
    plt.ylabel('Cumulative Energy Proportion (%)')
    plt.title('Energy Content of POD Modes')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    if r is not None:
        r_final = r
    else:
        r_final = int(np.argmax(cumulative_energy_proportion >= 100 * cumulative_energy_ratio) + 1)
    phi_scaled = U[:, :r_final]
    phi_scaled = np.ascontiguousarray(phi_scaled)

    # Plot the first up-to-10 POD mode shapes; each subplot's title shows the mode index
    # and its individual energy fraction (sigma_i^2 / sum(sigma^2)).
    n_modes_to_plot = min(10, r_final)
    fig, axes = plt.subplots(2, 5, figsize=(16, 5), sharex=True, constrained_layout=True)
    axes_flat = axes.flatten()
    total_energy = float(np.sum(S ** 2))
    for i in range(n_modes_to_plot):
        ax = axes_flat[i]
        ax.plot(phi_scaled[:, i], 'o-', ms=3, lw=1)
        ax.axhline(0, color='gray', lw=0.5)
        energy_pct = 100 * S[i] ** 2 / total_energy
        ax.set_title(f'mode {i+1}  ({energy_pct:.2f}%)', fontsize=10)
        ax.grid(True, alpha=0.3)
    for i in range(n_modes_to_plot, 10):
        axes_flat[i].axis('off')
    fig.supxlabel('node index')
    fig.supylabel('$\\phi_i$ (scaled)')
    fig.suptitle(f'First {n_modes_to_plot} POD modes (scaled coords)')
    plt.show()

    energy_proportion = 100 * np.sum(S[:r_final] ** 2) / np.sum(S ** 2)
    print(f"Energy proportion of the selected {r_final} modes: {energy_proportion:.6f}%")
    # test the relative RMSE = sqrt(1 - energy_proportion)
    Q_scaled_proj = phi_scaled @ (phi_scaled.T @ Q_scaled)
    err_proj = np.sqrt(
        np.mean(np.linalg.norm(Q_scaled - Q_scaled_proj, axis=0) ** 2) /
        np.mean(np.linalg.norm(Q_scaled, axis=0) ** 2)        
    )
    print(f"Square root of energy loss = {100*np.sqrt(1 - energy_proportion/100):.4f}%")
    print(f"Projection error: {100*err_proj:.4f}%")
    print(f"Projection error expected from Eckart-Young Theorem: {100*np.sqrt(1 - energy_proportion/100):.4f}%")
    
    return phi_scaled, r_final