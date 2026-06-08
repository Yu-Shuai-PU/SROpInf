"""sr_tool.py: utility tools for symmetry reduction procedure"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Callable, Optional, Tuple, Union
from scipy.integrate import solve_ivp
from sklearn.model_selection import KFold
from SROpInf.typing import Vector, Matrix, ROMTensorTuple
from SROpInf.models.model import (
    SymmetryReducedScaledFullOrderModel,
    SymmetryReducedScaledReducedOrderModel,
)
from SROpInf.grids.grid1d import Grid1DUniformSpectral, Grid1DCubicSpline

from itertools import combinations_with_replacement, permutations

def template_fitting(Q: Matrix,
                    grid: Union[Grid1DUniformSpectral, Grid1DCubicSpline],
                    q_template: Vector,
                    q_template_perp: Vector) -> Tuple[Matrix, Vector]:
    """Formula for calculating the shifting amount given template

    q_template = cos(2πx/L)
    T(L/4)q_template = cos(2π(x-L/4)/L) = sin(2πx/L)

    c(t) = argmin_c || shift_func(q(t), -c) - q_template ||_2
         = arg(<q(t), q_template> + 1j * <q(t), T(L/4)q_template>) * (L / 2π)

    returns:
    - Q_fitted: shifted snapshots, shape (n_grid, n_snapshots)
    - shift_amount: shift amounts c(t) for each snapshot, shape (n_snapshots, )
    """

    Q_fitted = np.zeros_like(Q)
    c = np.zeros(Q.shape[1], dtype=float)
    for idx_time in range(Q.shape[1]):
        q = Q[:, idx_time]
        c[idx_time] = np.angle(grid.inner_product(q, q_template) + 1j * grid.inner_product(q, q_template_perp)) * (grid.Lx / (2 * np.pi))
    c = np.unwrap(c * 2 * np.pi / grid.Lx) * (grid.Lx / (2 * np.pi)) # unwrap the shift amount to avoid artificial jumps due to periodicity
    Q_fitted = grid.shift_x(Q, -c)
    return Q_fitted, c

def _build_poly_data_matrix(Z: Matrix, poly_comp: list) -> Matrix:
    """Input Z, output [Z; Z⊗Z; ...],
    where the polynomial composition is determined by "poly_comp" list.
    Note that for the quadratic term Z⊗Z, we only include the unique monomials with replacement,
    i.e. if r=3 then we include [Z1^2, Z1*Z2, Z1*Z3, Z2^2, Z2*Z3, Z3^2]."""
    r, num_snapshot = Z.shape
    data_matrix_list = []
    current_power = np.ones((1, num_snapshot))
    last_d = 0
    for d in poly_comp:
        while last_d < d:
            current_power = np.einsum('it, jt -> ijt', Z, current_power).reshape(-1, num_snapshot)
            last_d += 1
        indices = list(combinations_with_replacement(range(r), d))
        flat_indices = [sum(val * (r**i) for i, val in enumerate(reversed(combo))) for combo in indices]
        data_matrix_list.append(current_power[flat_indices, :])
    return np.vstack(data_matrix_list)

def _build_opinf_poly_tensors(ops_poly: Matrix, r: int, poly_comp: list) -> ROMTensorTuple:
    """Input the duplicate-free reduced-order operators:
    ops_poly in shape (r, num_poly_terms), where num_poly_terms is the number of unique monomials with replacement determined by "poly_comp" list;
    ops = [A1_df, A2_df, ...],
    output [A1, A2], where A2_ijk = A2_ikj = A2_df_m,
    where m is the index of the monomial Z_i * Z_j in the duplicate-free list of monomials with replacement."""
    operators_list = []
    curr_col, output_dim = 0, ops_poly.shape[0]
    for d in poly_comp:
        unique_combos = list(combinations_with_replacement(range(r), d))
        num_feat = len(unique_combos)
        ops_block = ops_poly[:, curr_col : curr_col + num_feat]
        curr_col += num_feat

        full_tensor = np.zeros((output_dim,) + tuple([r] * d))
        for idx_feat, combo in enumerate(unique_combos):
            perms = set(permutations(combo))
            share_val = ops_block[:, idx_feat] / len(perms)
            for perm in perms:
                full_tensor[(slice(None),) + perm] = share_val

        if output_dim == 1: full_tensor = full_tensor.squeeze(axis = 0)
        operators_list.append(full_tensor)
    return tuple(operators_list)

def _assemble_normal_equations(
    Z: Matrix,                 # latent snapshots, shape (r, n_cols)
    fQ_latent: Matrix,         # Phi^T f(Q_fitted), shape (r, n_cols)
    dcdt_numer: Vector,        # -<q_template_dx, f(Q_fitted)>, shape (n_cols,)
    poly_comp: list,
) -> Tuple[Matrix, Matrix, Vector, int]:
    """Form the shared Gram and the two RHS of the decoupled ridge regressions from already-assembled
    latent data. Both regressions share the polynomial data matrix P = [Z; Z⊗Z; ...]:
        G      = P P^T          (n_poly, n_poly)
        B_poly = P fQ_latent^T  (n_poly, r)      -> target Phi^T f
        B_dcdt = P dcdt_numer   (n_poly,)        -> target -<f, q_template_dx>
    These are SUMMED over the n_cols snapshots (not yet normalized), so they are additive across data
    subsets: a K-fold training fold is assembled by simply summing the per-trajectory pieces, and the
    per-snapshot normalization is deferred to _solve_normal_eqs. Returns (G, B_poly, B_dcdt, n_cols)."""
    P = _build_poly_data_matrix(Z, poly_comp)            # (n_poly, n_cols)
    G = P @ P.T                                          # (n_poly, n_poly)
    B_poly = P @ fQ_latent.T                             # (n_poly, r)
    B_dcdt = P @ dcdt_numer                              # (n_poly,)
    return G, B_poly, B_dcdt, P.shape[1]

def _solve_normal_equations(
    G: Matrix, B_poly: Matrix, B_dcdt: Vector, n_cols: int,
    poly_comp: list, r: int,
    lam_poly: float, lam_dcdt: float
) -> ROMTensorTuple:
    """Solve the two decoupled ridge regressions from pre-assembled normal equations and assemble the
    ROM tensor tuple in the same layout as SymmetryReducedScaledFullOrderModel.project
    (poly rhs tensors, then shift-speed-numerator tensors, then the basis-only dcdt-denominator and
    advection operators).

    The Gram and both RHS are normalized by n_cols (a per-snapshot AVERAGE rather than a sum), so the
    regularizers act on a data-size-independent scale: a lambda chosen during K-fold CV (on N_train
    snapshots) is directly reusable when retraining on all N_all snapshots -- no rescaling needed.
    At lambda = 0 the 1/n_cols cancels, so the unregularized re-projection solve still recovers the
    S-R POD-Galerkin operators exactly."""
    n_poly = G.shape[0]
    G_n = G / n_cols
    ops_poly = np.linalg.solve(G_n + lam_poly * np.eye(n_poly), B_poly / n_cols).T   # (r, n_poly)
    ops_dcdt = np.linalg.solve(G_n + lam_dcdt * np.eye(n_poly), B_dcdt / n_cols)     # (n_poly,)
    return _build_opinf_poly_tensors(ops_poly, r, poly_comp) \
         + _build_opinf_poly_tensors(ops_dcdt.reshape(1, -1), r, poly_comp)

def _traj_relative_RMSE(
    rom: SymmetryReducedScaledReducedOrderModel,
    Q_fitted_fom: Matrix,     # held-out FOM fitted snapshots, shape (n_grid, n_snapshots)
    c_fom: Vector,        # held-out FOM shift amounts c(t), shape (n_snapshots,)
    Phi: Matrix,
    t_eval: Vector,
) -> float:
    """Trajectory relative RMSE of the fold ROM on ONE held-out trajectory, scored in the LAB
    (un-shifted, physical) frame -- i.e. on the snapshots themselves, NOT the fitted snapshots.

    Both the ROM reconstruction and the FOM reference are shifted back to the physical frame by
    their respective shift amounts before the comparison, so the metric penalizes the bulk-
    translation (location) error in addition to the shape/amplitude error -- matching the lab-frame
    relative error reported by sample_and_compare.

    The ROM state is [z, c]; it is integrated from y0 = [Phi^T Q_fitted_traj[:, 0], c_fom_traj[0]]
    (the TRUE initial shift, which matters in the lab frame). The reconstruction Phi @ z(t) is then
    un-shifted by the ROM-predicted c(t), and the FOM fitted snapshots by the recorded c_fom(t)
    (recovering the original lab snapshots). This is a genuine time-integrated trajectory metric,
    NOT a regression residual.

    A diverged solve makes this RMSE NaN (rom.solve NaN-pads the post-blow-up tail, and np.mean
    over a NaN-containing column is NaN), so the (lambda) pair's CV score is NaN and is excluded by
    the np.nanargmin selection / the all-finite guard.
    """
    z0 = Phi.T @ Q_fitted_fom[:, 0]                          # the initial latent state obtained by projecting the fitted initial snapshot onto the S-R POD basis
    
    y = rom.solve(
        z0 = z0,
        c0 = c_fom[0],
        t_eval = t_eval
    )

    z_rom, c_rom = y[:-1], y[-1]
    Q_rom = rom.shift_x(Phi @ z_rom, c_rom)
    Q_fom = rom.shift_x(Q_fitted_fom, c_fom)
    return float(np.sqrt(np.mean(np.linalg.norm(Q_rom - Q_fom, axis=0) ** 2) / np.mean(np.linalg.norm(Q_fom, axis=0) ** 2)))

def _plot_cv_grid_search(
    cv_score: Matrix,          # (n_poly_sweep, n_dcdt_sweep) mean held-out relative RMSE (fraction)
    reg_poly_range: Vector,
    reg_dcdt_range: Vector,
    best_p: int, best_d: int,
    fig_path: str,
) -> None:
    """2D heatmap (log-log) of the held-out trajectory relative RMSE over the (lambda_poly,
    lambda_dcdt) grid; the CV-optimal pair is starred. Diverged (+inf) pairs are masked out."""
    plt.figure(figsize=(8, 6))
    data = np.ma.masked_invalid(100 * cv_score.T)   # (n_dcdt, n_poly), percent units
    # log color scale: held-out RMSE spans orders of magnitude across the (lambda) grid; diverged
    # (nan-masked) pairs render blank. clip vmin to the smallest finite positive entry for LogNorm.
    finite = data.compressed()
    vmin = float(finite[finite > 0].min()) if finite.size and (finite > 0).any() else None
    pcm = plt.pcolormesh(reg_poly_range, reg_dcdt_range, data, shading="nearest", cmap="viridis",
                         norm=LogNorm(vmin=vmin, vmax=float(finite.max()) if finite.size else None))
    plt.colorbar(pcm, label="held-out relative RMSE (%, log scale)")
    plt.plot(reg_poly_range[best_p], reg_dcdt_range[best_d], marker="*",
             color="red", markersize=18, markeredgecolor="black")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r"$\lambda_{\mathrm{poly}}$")
    plt.ylabel(r"$\lambda_{\mathrm{dcdt\,numer}}$")
    plt.title(rf"S-R OpInf {len(reg_poly_range)}$\times${len(reg_dcdt_range)} grid search "
              rf"(best $\lambda_p$={reg_poly_range[best_p]:.1e}, $\lambda_d$={reg_dcdt_range[best_d]:.1e}, "
              rf"{100*cv_score[best_p, best_d]:.2f}%)")
    plt.tight_layout()
    plt.savefig(fig_path + "sropinf_cv_grid_search.png", dpi=300)
    plt.close()

def sropinf(
    poly_comp: list, # the polynomial composition of the S-R FOM dynamics; for example, if it's [1, 2] then we include both linear and quadratic terms in the S-R FOM dynamics; if it's [2] then we only include the quadratic term in the S-R FOM dynamics
    Phi: Matrix, # the S-R POD basis matrix Phi, shape (n_grid, r)
    Q_fitted: np.ndarray, # the shifted snapshot matrix Q_fitted, shape (n_grid, n_snapshots) or (n_grid, n_snapshots, n_trajs) if we train on multiple trajectories with random perturbations
    fom_sr: SymmetryReducedScaledFullOrderModel, # the symmetry-reduced full-order model for us to obtain the right-hand side velocity vector
    re_projection_option: bool = False, # whether to perform the re-projection step (evaluate f at the lifted state Phi Phi^T Q_fitted); requires zero regularization and recovers the S-R POD-Galerkin operators exactly. Incompatible with grid_search_option.
    grid_search_option: bool = False, # if True, treat both regularizers as 1D arrays and run a 2D Cartesian grid search (scoring set by cross_validation_option); if False, do a single-shot solve at the given (scalar) regularizer pair
    cross_validation_option: bool = False, # only used when grid_search_option is True: if True, score each regularizer pair by num_split-fold cross-validation over trajectories (held-out; requires n_trajs >= num_split); if False, score in-sample on all data (allows a single trajectory)
    penalty_weight_rhs_poly: Union[float, Vector] = 0.0, # scalar regularizer for the polynomial part (grid_search_option False), or 1D array of candidates (grid_search_option True)
    penalty_weight_dcdt_numer: Union[float, Vector] = 0.0, # scalar regularizer for the dcdt-numerator part (grid_search_option False), or 1D array of candidates (grid_search_option True)
    t_eval: Optional[Vector] = None, # the time grid for the trajectory-RMSE scoring (e.g. params.tsave); REQUIRED when grid_search_option is True, ignored otherwise
    shift_amount: Optional[np.ndarray] = None, # per-trajectory FOM shift amounts c(t), shape (n_snapshots, n_trajs) (or (n_snapshots,) for a single trajectory); REQUIRED when grid_search_option is True (used to un-shift both the ROM reconstruction and the FOM reference back to the lab frame, so the scoring metric is the relative RMSE on the snapshots, not the fitted snapshots)
    num_split: int = 10, # number of K-folds over trajectories in the CV branch
    opinf_CV_random_seed: int = 42, # KFold shuffle seed
    fig_path: Optional[str] = None, # if given, save the 2D CV grid-search heatmap to this directory
) -> ROMTensorTuple:
    """Method to solve the S-R OpInf problem and obtain the reduced-order operators.

    The minimization problems are:
        min_{A, B} || Phi^T f(Q_fitted) - (A Z + B (Z ⊗ Z)) ||_F^2 + lambda_poly * (|A|_F^2 + |B|_F^2), where f is the right-hand side vector field of the original FOM, as well as the polynomial part of the S-R FOM dynamics
        min_{p, Q} || -<f(Q_fitted), dq_template_dx> - (p^T Z + Z^T Q Z) ||_2^2 + lambda_dcdt_numer * (|p|_2^2 + |Q|_F^2)

    If both regularizers are 0, and the "re_projection_option" is True, then the above minimization problems exactly recover the reduced-order operators of the S-R POD-Galerkin ROM.

    Three execution modes:
      - grid_search_option=False: single-shot solve at the given (scalar) regularizer pair. With
        re_projection_option=True and zero regularizers this recovers the S-R POD-Galerkin ROM.
      - grid_search_option=True, cross_validation_option=False: 2D (lambda_poly x lambda_dcdt) grid
        search scored IN-SAMPLE -- each pair is trained on ALL data and scored by the mean trajectory
        relative RMSE on that same data. Works for a single trajectory and is cheaper, but optimistic
        (no held-out set).
      - grid_search_option=True, cross_validation_option=True: same grid, scored by num_split-fold
        cross-validation OVER TRAJECTORIES (requires n_trajs >= num_split): the mean held-out
        trajectory relative RMSE.
      Both grid-search modes score by a genuine time-integrated trajectory RMSE (not a regression
      residual) and (re)train the final operators on ALL trajectories at the selected pair; the
      Gram/RHS are per-snapshot normalized so the selected lambda transfers without rescaling.
    """

    assert Q_fitted.ndim in [2, 3], "The shifted snapshot matrix 'Q_fitted' should be either 2D (n_grid, n_snapshots) or 3D (n_grid, n_snapshots, n_trajs)"

    if re_projection_option and grid_search_option:
        raise ValueError("Re-projection is incompatible with grid-search cross-validation: re-projection "
                         "requires zero regularization (it recovers the Galerkin ROM), while a grid search "
                         "tunes nonzero regularizers.")
    if re_projection_option and (np.any(np.asarray(penalty_weight_rhs_poly) != 0.0)
                                 or np.any(np.asarray(penalty_weight_dcdt_numer) != 0.0)):
        raise ValueError("Cannot use regularization while also performing the re-projection step in the S-R OpInf procedure")

    r = Phi.shape[1] # the dimension of the S-R POD subspace

    # 1. get known (basis-only) tensors, shared by both modes:
    dcdt_denom_linear = fom_sr.shift_speed_denom(Phi)
    advection_linear = Phi.T @ fom_sr.advection(Phi)

    # 2a. Single-shot solve at the given scalar regularizer pair (re-projection optional).
    if not grid_search_option:
        Q_fitted = Q_fitted.reshape(Q_fitted.shape[0], -1)  # (n_grid, n_snap * n_traj); flattens the trajectory axis if present
        Z = Phi.T @ Q_fitted
        if re_projection_option:
            # re-project: evaluate f at the lifted state Phi Phi^T Q_fitted (removes the off-subspace component)
            Q_fitted = Phi @ Z
        fQ_fitted = fom_sr.rhs_poly(Q_fitted)
        PhiT_fQ_fitted = Phi.T @ fQ_fitted
        dcdt_numer = -np.dot(fom_sr.q_template_dx_scaled, fQ_fitted)
        G, B_poly, B_dcdt, n_cols = _assemble_normal_equations(Z, PhiT_fQ_fitted, dcdt_numer, poly_comp)
        
        tensors_trainable = _solve_normal_equations(G, B_poly, B_dcdt, n_cols, poly_comp, r,
                                 float(penalty_weight_rhs_poly), float(penalty_weight_dcdt_numer))
        tensors_sropinf = tensors_trainable + (dcdt_denom_linear, advection_linear)
        return tensors_sropinf

    # 2b. Grid search over a (lambda_poly x lambda_dcdt) grid. Each pair is scored by the MEAN
    #     trajectory relative RMSE (a time-integrated metric, NOT a regression residual): in-sample on
    #     all data (cross_validation_option=False) or held-out via K-fold CV over trajectories
    #     (cross_validation_option=True). The final operators are (re)trained on ALL data at the
    #     selected pair; per-snapshot normalization makes the selected lambda transfer without rescaling.
    if t_eval is None:
        raise ValueError("grid_search_option=True requires `t_eval` (e.g. params.tsave) for the trajectory-RMSE validation metric.")
    if shift_amount is None:
        raise ValueError("grid_search_option=True requires `shift_amount` (per-trajectory FOM shift amounts c(t)) to move fitted FOM snapshots back to the lab frame for the relative-RMSE validation.")

    # Unify to a trajectory axis so the single-trajectory (in-sample) and multi-trajectory cases share
    # one code path: Q_fitted -> (n_grid, n_snapshots, n_trajs), shift_amount -> (n_snapshots, n_trajs).
    if Q_fitted.ndim == 2:
        Q_fitted = Q_fitted[:, :, None]
    shift_amount = np.asarray(shift_amount)
    if shift_amount.ndim == 1:
        shift_amount = shift_amount[:, None]
    n_traj = Q_fitted.shape[2]

    if cross_validation_option and n_traj < num_split:
        raise ValueError(
            f"cross_validation_option=True needs n_trajs >= num_split (={num_split}) for K-fold CV "
            f"over trajectories; got n_trajs = {n_traj}. Use multiple training trajectories, or set "
            f"cross_validation_option=False to score in-sample on all data (works for a single trajectory)."
        )

    penalty_weight_rhs_poly_range = np.atleast_1d(np.asarray(penalty_weight_rhs_poly, dtype=float))
    penalty_weight_dcdt_numer_range = np.atleast_1d(np.asarray(penalty_weight_dcdt_numer, dtype=float))

    # --- Precompute per-trajectory normal equations ONCE (heavy f(Q_fitted) evals; NO re-projection,
    #     since the CV/regularized path is the practical non-Galerkin ROM). Pieces are additive. ---
    G_pt, Bpoly_pt, Bdcdt_pt, ncol_pt = [], [], [], []
    for j in range(n_traj):
        Qj = Q_fitted[:, :, j]
        Zj = Phi.T @ Qj
        fQj = fom_sr.rhs_poly(Qj)
        Gj, Bpj, Bdj, nj = _assemble_normal_equations(
            Zj, Phi.T @ fQj, -np.dot(fom_sr.q_template_dx_scaled, fQj), poly_comp)
        G_pt.append(Gj); Bpoly_pt.append(Bpj); Bdcdt_pt.append(Bdj); ncol_pt.append(nj)
    G_all = sum(G_pt); Bp_all = sum(Bpoly_pt); Bd_all = sum(Bdcdt_pt); nc_all = sum(ncol_pt)

    # Build a ROM from pre-assembled (summed) normal equations at a given regularizer pair.
    def _build_rom(G, Bp, Bd, nc, lam_poly, lam_dcdt):
        tensors_trainable = _solve_normal_equations(G, Bp, Bd, nc, poly_comp, r, lam_poly, lam_dcdt)
        return SymmetryReducedScaledReducedOrderModel.build(
            poly_comp=poly_comp,
            phi_scaled=Phi,
            tensors=tensors_trainable + (dcdt_denom_linear, advection_linear),
            q_template_dxx_scaled=fom_sr.q_template_dxx_scaled,
            shift_func=fom_sr.shift_x,
            shift_speed_denom_threshold=0.0, # during the offline training phase, we don't use regularized shift speed denominator
        )

    # --- K-fold split (held-out CV only); pre-sum each fold's TRAINING normal equations once
    #     (regularizer-independent), so each (pair, fold) only re-solves a small (n_poly x n_poly) system. ---
    if cross_validation_option:
        kfold = KFold(n_splits=num_split, shuffle=True, random_state=opinf_CV_random_seed)
        fold_splits = list(kfold.split(np.arange(n_traj)))
        fold_train_eqs = [(sum(G_pt[j] for j in tr), sum(Bpoly_pt[j] for j in tr),
                           sum(Bdcdt_pt[j] for j in tr), sum(ncol_pt[j] for j in tr))
                          for (tr, _) in fold_splits]

    # Score one regularizer pair by the mean trajectory relative RMSE: held-out over CV folds, or
    # in-sample over all trajectories (train == val == all data).
    def _score(lam_poly, lam_dcdt):
        errors = []
        if cross_validation_option:
            for (_, val_idx), (G, Bp, Bd, nc) in zip(fold_splits, fold_train_eqs):
                rom = _build_rom(G, Bp, Bd, nc, lam_poly, lam_dcdt)
                for j in val_idx:
                    errors.append(_traj_relative_RMSE(rom, Q_fitted[:, :, j], shift_amount[:, j], Phi, t_eval))
        else:
            rom = _build_rom(G_all, Bp_all, Bd_all, nc_all, lam_poly, lam_dcdt)
            for j in range(n_traj):
                errors.append(_traj_relative_RMSE(rom, Q_fitted[:, :, j], shift_amount[:, j], Phi, t_eval))
        return float(np.mean(errors))

    score_kind = "held-out" if cross_validation_option else "in-sample"
    cv_score = np.full((len(penalty_weight_rhs_poly_range), len(penalty_weight_dcdt_numer_range)), np.nan)
    for idx_p, lam_rhs_poly in enumerate(penalty_weight_rhs_poly_range):
        for idx_d, lam_dcdt in enumerate(penalty_weight_dcdt_numer_range):
            cv_score[idx_p, idx_d] = _score(lam_rhs_poly, lam_dcdt)
            print(f"lambda_rhs_poly = {lam_rhs_poly:.2e}, lambda_dcdt = {lam_dcdt:.2e}: "
                  f"mean {score_kind} relative RMSE = {100 * cv_score[idx_p, idx_d]:.2f}%")

    # --- Select the argmin pair, plot the heatmap, (re)train on ALL trajectories at that pair ---
    if not np.any(np.isfinite(cv_score)):
        raise RuntimeError("Every (lambda_rhs_poly, lambda_dcdt) pair diverged / produced non-finite scores; "
                           "widen or shift the regularizer ranges toward larger values.")
    best_p, best_d = np.unravel_index(np.nanargmin(cv_score), cv_score.shape)
    print(f"Selected lambda_rhs_poly = {penalty_weight_rhs_poly_range[best_p]:.2e}, lambda_dcdt = {penalty_weight_dcdt_numer_range[best_d]:.2e} "
          f"({score_kind} relative RMSE = {100 * cv_score[best_p, best_d]:.2f}%)")
    if fig_path is not None:
        _plot_cv_grid_search(cv_score, penalty_weight_rhs_poly_range, penalty_weight_dcdt_numer_range, best_p, best_d, fig_path)

    tensors_trainable = _solve_normal_equations(G_all, Bp_all, Bd_all, nc_all, poly_comp, r,
                             float(penalty_weight_rhs_poly_range[best_p]), float(penalty_weight_dcdt_numer_range[best_d]))
    tensors_sropinf = tensors_trainable + (dcdt_denom_linear, advection_linear)

    return tensors_sropinf
