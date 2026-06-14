import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.ticker import PercentFormatter, FuncFormatter
from matplotlib.colors import to_rgb

from typing import Any, Dict, List, Callable, Union
from SROpInf.typing import Vector, Matrix
from SROpInf.grids.grid1d import Grid1DUniformSpectral, Grid1DCubicSpline, fft

def _on_white(color, alpha: float):
    """Pre-blend `color` at opacity `alpha` over a white background. The PostScript/EPS backend
    does not support transparency (alpha-carrying artists are rendered fully opaque), so any
    translucent fill must be baked into the equivalent solid color before saving to .eps."""
    r, g, b = to_rgb(color)
    return (1 - alpha * (1 - r), 1 - alpha * (1 - g), 1 - alpha * (1 - b))

def _spectral_upsample_field(Q: Matrix, nx_plot: int) -> Matrix:
    """Fourier zero-pad a periodic real field Q (shape (nx, nt)) along the space axis up to nx_plot
    points: lossless for a spectrally-resolved solution (its Fourier content is unchanged, only the
    plotting grid is refined), so it removes the faceting a coarse-nx contourf shows without altering
    the data. The rfft is per-column, so NaN-padded columns of a diverged ROM stay NaN. Returns Q
    unchanged when nx_plot is None or nx_plot <= nx. The (nx_plot/nx) factor offsets irfft's 1/n
    normalization so amplitudes are preserved."""
    nx = Q.shape[0]
    if nx_plot is None or nx_plot <= nx:
        return Q
    Uhat = np.fft.rfft(Q, axis=0)
    pad = np.zeros((nx_plot // 2 + 1, Q.shape[1]), dtype=complex)
    pad[:Uhat.shape[0]] = Uhat
    return np.fft.irfft(pad, n=nx_plot, axis=0) * (nx_plot / nx)

def _interp_time_valid(t_src: Vector, arr: np.ndarray, t_query: Vector, kind: str = "cubic") -> np.ndarray:
    """Interpolate `arr` (shape (n_state, n_t) or (n_t,)) from `t_src` onto `t_query` along the time
    axis, robust to a NaN-padded diverged tail. A blown-up ROM is NaN-padded past blow-up, and a
    global cubic spline turns a single NaN into an all-NaN curve; so here we interpolate only over
    the finite time samples and leave `t_query` outside their span NaN -- the pre-divergence segment
    still renders. Falls back to linear when fewer than 4 finite samples remain."""
    is2d = arr.ndim == 2
    finite = np.all(np.isfinite(arr), axis=0) if is2d else np.isfinite(arr)
    if finite.all():
        return interp1d(t_src, arr, axis=-1, kind=kind)(t_query)
    out = np.full((arr.shape[0], t_query.size) if is2d else (t_query.size,), np.nan)
    t_ok = t_src[finite]
    if t_ok.size < 2:
        return out
    arr_ok = arr[:, finite] if is2d else arr[finite]
    k = kind if t_ok.size >= 4 else "linear"
    in_range = (t_query >= t_ok[0]) & (t_query <= t_ok[-1])
    vals = interp1d(t_ok, arr_ok, axis=-1, kind=k)(t_query[in_range])
    if is2d:
        out[:, in_range] = vals
    else:
        out[in_range] = vals
    return out

def plot_traj_contourf_xt(
    Q_fom: Matrix,
    Q_rom: Matrix,
    error_rom: float,
    x: Vector,
    t: Vector,
    model_name_rom: str,
    model_abbrev_rom: str,
    fig_path: str,
    idx_traj: int,
    nx_plot: int = 256
):
    # Fourier zero-pad the space axis to nx_plot points so a coarse-nx field renders with smooth
    # contourf edges instead of facets (lossless for the spectral solution); no-op when nx >= nx_plot.
    if nx_plot is not None and Q_fom.shape[0] < nx_plot:
        Lx = x[-1] + (x[1] - x[0])
        x = np.linspace(0.0, Lx, nx_plot, endpoint=False)
        Q_fom = _spectral_upsample_field(Q_fom, nx_plot)
        Q_rom = _spectral_upsample_field(Q_rom, nx_plot)

    # Symmetric colour limits rounded UP to an integer so the colorbar limits/ticks are integers
    # (nan-aware: a blown-up ROM is NaN-padded past divergence); shared across the FOM / ROM /
    # difference panels so the three are directly comparable.
    
    # vlim = float(np.ceil(np.nanmax(np.abs(np.concatenate([Q_fom, Q_rom])))))
    vlim = 15
    # levels = np.linspace(-vlim, vlim, 51)
    levels = np.linspace(-vlim, vlim, 21)
    
    # integer colorbar ticks; fall back to matplotlib's default when a divergent ROM makes vlim
    # large enough that an explicit integer tick array would be unreadable.
    cbar_ticks = np.arange(-vlim, vlim + 1, 1) if vlim <= 10 else None
    # periodic grid omits the endpoint x = Lx; recover it so the 2*pi tick sits at the right edge.
    Lx = x[-1] + (x[1] - x[0])
    xticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    xticklabels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    plt.figure(figsize=(10, 6))
    pcm = plt.contourf(x, t, Q_fom.T, levels=levels, cmap='RdBu_r')
    # cbar = plt.colorbar(pcm, ticks=cbar_ticks)
    cbar = plt.colorbar(pcm, ticks=[-15, -10, -5, 0, 5, 10, 15])
    cbar.ax.tick_params(labelsize=16)
    plt.xticks(xticks, xticklabels)
    plt.xlim(0, Lx)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(fig_path + f"traj_{idx_traj:03d}_fom.png", dpi=300)
    plt.savefig(fig_path + f"traj_{idx_traj:03d}_fom.eps", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    pcm = plt.contourf(x, t, Q_rom.T, levels=levels, cmap='RdBu_r')
    # cbar = plt.colorbar(pcm, ticks=cbar_ticks)
    cbar = plt.colorbar(pcm, ticks=[-15, -10, -5, 0, 5, 10, 15])
    cbar.ax.tick_params(labelsize=16)
    # plt.xlabel(r"$x$", fontsize=20)
    # plt.ylabel(r"$t$", fontsize=20)
    # plt.title(rf"Trajectory {idx_traj:03d}, {model_name_rom}, relative RMSE = {100*error_rom:.2f}%", fontsize=18) # type: ignore
    plt.xticks(xticks, xticklabels)
    plt.xlim(0, Lx)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(fig_path + f"traj_{idx_traj:03d}_{model_abbrev_rom}.png", dpi=300)
    plt.savefig(fig_path + f"traj_{idx_traj:03d}_{model_abbrev_rom}.eps", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    pcm = plt.contourf(x, t, Q_fom.T - Q_rom.T, levels=levels, cmap='RdBu_r')
    cbar = plt.colorbar(pcm, ticks=cbar_ticks)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel(r"$x$", fontsize=20)
    plt.ylabel(r"$t$", fontsize=20)
    plt.title(rf"Trajectory {idx_traj:03d}, FOM - {model_name_rom} diff, relative RMSE = {100*error_rom:.2f}%", fontsize=18) # type: ignore
    plt.xticks(xticks, xticklabels)
    plt.xlim(0, Lx)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(fig_path + f"traj_{idx_traj:03d}_diff_fom_{model_abbrev_rom}.png", dpi=300)
    plt.close()

def plot_traj_x(
    Q_list: List[Matrix],
    error_list: List[float],
    x: Vector,
    t: Vector,
    model_name_list: List[str],
    color_list: List[str],
    linestyle_list: List[str],
    fig_path: str,
    idx_traj: int
):
    
    # Q_list = [Q_fom, Q_rom_1, ..., Q_rom_K]                 → len = K+1
    # error_list = [error_rom_1, ..., error_rom_K]            → len = K   (FOM has no self-error)
    # error for idx_model >= 1 lives at error_list[idx_model - 1].
    for idx_time in range(len(t)):
        plt.figure(figsize=(12, 6))
        for idx_model in range(len(Q_list)):
            if idx_model == 0:
                label = rf"{model_name_list[idx_model]}"  # FOM: no error
            else:
                label = rf"{model_name_list[idx_model]}, relative RMSE = {100 * error_list[idx_model - 1]:.2f}%"
            plt.plot(x, Q_list[idx_model][:, idx_time],
                     label=label,
                     color=color_list[idx_model], linestyle=linestyle_list[idx_model])
        plt.xlabel(r"$x$", fontsize=20)
        plt.ylabel(rf"$q(x,t = {t[idx_time]:.2f})$", fontsize=20)
        plt.title(rf"Trajectory {idx_traj:03d} at $t = {t[idx_time]:.2f}$", fontsize=18)
        plt.tick_params(labelsize=16)
        plt.legend(loc = "upper right", fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_path + f"traj_{idx_traj:03d}_t_{int(t[idx_time])}.png", dpi=300)
        plt.close()
    
def plot_shift_t(
    shift_amount_list: List[Vector],
    inv_shift_speed_denom_list: List[Vector],
    time: Vector,
    model_name_list: List[str],
    color_list: List[str],
    linestyle_list: List[str],
    fig_path: str,
    idx_traj: int
):
    plt.figure(figsize=(10, 6))                      # a 10x6 aspect reads better than a square 10x10 for a line plot
    for idx_model in range(len(shift_amount_list)):
        plt.plot(time, shift_amount_list[idx_model], label=rf"{model_name_list[idx_model]}",
                color=color_list[idx_model], linestyle=linestyle_list[idx_model], lw=2)
    plt.xlim(time[0], time[-1])
    xt = np.arange(time[0], time[-1] + 1e-12, 2)
    plt.xticks(xt, [rf"${v:.0f}$" for v in xt], fontsize=16)
    yt = [-3.5, -3.0, -2.5, -2.0]
    plt.yticks(yt, [rf"${v}$" for v in yt], fontsize=16)
    # plt.xlabel(r"$t$", fontsize=20)
    # plt.ylabel(r"$c(t)$", fontsize=20)
    plt.tick_params(labelsize=16)                    
    plt.grid(True, color=_on_white("#b0b0b0", 0.3))
    # framealpha=1: the default 0.8 legend frame is a transparent artist the EPS backend cannot render
    plt.legend(loc="upper right", fontsize=16, framealpha=1.0)
    plt.tight_layout()
    plt.savefig(fig_path + f"shift_amount_{idx_traj:03d}.png", dpi=300)
    plt.savefig(fig_path + f"shift_amount_{idx_traj:03d}.eps", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    for idx_model in range(len(inv_shift_speed_denom_list)):
        plt.semilogy(time, inv_shift_speed_denom_list[idx_model], label=rf"{model_name_list[idx_model]}", color=color_list[idx_model], linestyle=linestyle_list[idx_model])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$1/D$")
    plt.ylim(4.5e-1, 6e-1)
    plt.title(rf"Trajectory {idx_traj:03d} inverse shift speed denominator $1/D$ (log-scaled) over time")
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.savefig(fig_path + f"inv_shift_speed_denom_{idx_traj:03d}_log_scaled.png", dpi=300)
    plt.close()

def plot_error_phase_amplitude_t(
    Q_list: List[Matrix],
    shift_amount_list: List[Vector],
    shift_func: Callable[[Matrix, Vector], Matrix],
    time: Vector,
    model_name_list: List[str],
    model_abbrev_list: List[str],
    fig_path: str,
    idx_traj: int
):
    """This function is for plotting the error in phase and amplitude, separately, to isolate phase mismatch with amplitude mismatch
    input: u_FOM and u_ROM, both of shape (n_state, n_t)
        - first compute u_FOM = sum_k a_k_FOM(t) exp(1j * k * x); u_ROM = sum_k a_k_ROM(t) exp(1j * k * x)
            where a_k is given by our fft function in grid1d.py
        
        - next decompose a_k_FOM(t) = A_k_FOM(t) exp(1j * phi_k_FOM(t)); a_k_ROM(t) = A_k_ROM(t) exp(1j * phi_k_ROM(t))
        
        - then compute error in amplitude: A_k_FOM(t) - A_k_ROM(t); error in phase: phi_k_FOM(t) - phi_k_ROM(t)
        
        - is there any relationship between |u_FOM - u_ROM|^2 (space-time L2 norm) and int_0^T sum_k |A_k_FOM(t) - A_k_ROM(t)|^2 and int_0^T sum_k |phi_k_FOM(t) - phi_k_ROM(t)|^2 dt? Can we attribute the error in trajectory to error in amplitude vs error in phase? 
        - is there any relationship between int_0^T sum_k |A_k_FOM(t) - A_k_ROM(t)|^2 and |u_FOM_fitted_by_c - u_ROM_fitted_by_c_rom|^2 (space-time L2 norm)?
        - is there any relationship between int_0^T sum_k |phi_k_FOM(t) - phi_k_ROM(t)|^2 and int_0^T |c - c_rom|^2?
        
        Answer to the Question 1:
            1. From the Parseval's theorem, we know that:
                int_0^T |u_FOM - u_ROM|^2_2 dt = int_0^T sum_k |a_k_FOM(t) - a_k_ROM(t)|^2 dt
                ||_2^2 is given by the inner product function in grid1d.py
            2. |a_k_FOM(t) - a_k_ROM(t)|^2 = |A_k_FOM(t) - A_k_ROM(t)|^2 + 2 * A_k_FOM(t) * A_k_ROM(t) * (1 - cos(phi_k_FOM(t) - phi_k_ROM(t)))
                (this is because |a - a'|^2 = A^2 + A'^2 - 2 A A' cos(phi - phi') = (A - A')^2 + 2 A A' (1 - cos(phi - phi')))
            3. Therefore, 
                int_0^T |u_FOM - u_ROM|^2_2 dt = int_0^T sum_k |A_k_FOM(t) - A_k_ROM(t)|^2 dt + int_0^T sum_k (2 * A_k_FOM(t) * A_k_ROM(t)) * (1 - cos(phi_k_FOM(t) - phi_k_ROM(t))) dt
        
        Answer to the Question 2:
            1. u_FOM_fitted_by_c = sum_k a_k_FOM(t) exp(1j * k * (x + c(t))) = sum_k a_k_FOM(t) * exp(1j * k * c(t)) * exp(1j * k * x)
               u_ROM_fitted_by_c_rom = sum_k a_k_ROM(t) exp(1j * k * (x + c_rom(t))) = sum_k a_k_ROM(t) * exp(1j * k * c_rom(t)) * exp(1j * k * x)
            2. int_0^T ||u_FOM_fitted_by_c - u_ROM_fitted_by_c_rom||^2_2 dt
                = int_0^T sum_k |a_k_FOM(t) * exp(1j * k * c(t)) - a_k_ROM(t) * exp(1j * k * c_rom(t))|^2 dt
                = int_0^T sum_k |A_k_FOM(t) * exp(1j * k * c(t) + 1j * phi_k_FOM(t)) - A_k_ROM(t) * exp(1j * k * c_rom(t) + 1j * phi_k_ROM(t))|^2 dt
                = int_0^T sum_k |A_k_FOM(t) - A_k_ROM(t)|^2 dt + int_0^T sum_k (2 * A_k_FOM(t) * A_k_ROM(t)) * (1 - cos( (phi_k_FOM(t) + k * c(t)) - (phi_k_ROM(t) + k * c_rom(t)))) dt
                
        Answer to the Question 3:
            It seems that int_0^T sum_k |phi_k_FOM(t) - phi_k_ROM(t)|^2 dt is not directly related to int_0^T |c - c_rom|^2 dt,
            because the phase error can come from both the shift amount mismatch and the intrinsic phase mismatch.
            
        Summary:
            1. Amplitude mismatch of spatial Fourier modes: int_0^T sum_k |A_k_FOM(t) - A_k_ROM(t)|^2 dt --- does ROM accurately compute the "energy" of each spatial scale?
            2. Intrinsic Phase mismatch of spatial Fourier modes: int_0^T sum_k |phi_k_FOM(t) + k * c(t) - (phi_k_ROM(t) + k * c_rom(t))|^2 dt --- does ROM accurately compute the "shape" of the wave packet?
            3. Shift amount mismatch: int_0^T |c - c_rom|^2 dt                                           --- does ROM accurately compute the "location" of the wave packet?
            4. total L2 error:
                int_0^T ||u_FOM - u_ROM||^2_2 dt
              = int_0^T sum_k |A_k_FOM(t) - A_k_ROM(t)|^2 dt
              + int_0^T sum_k (2 * A_k_FOM(t) * A_k_ROM(t)) * (1 - cos(phi_k_FOM(t) - phi_k_ROM(t))) dt (combining shift amount mismatch and intrinsic phase mismatch)

    Implementation: two amplitude/phase composition figures per ROM, both pointwise in time and
    normalized by the instantaneous FOM energy ||u_F(t)||_2^2 = sum_k |a_k^F|^2 (Parseval, exact on
    the uniform spectral grid). Each figure stacks amplitude error sum_k (A_k^F - A_k^R)^2 and phase
    error sum_k 2 A_k^F A_k^R (1 - cos d_phi); the two bands sum exactly to the total
    sum_k |a_k^F - a_k^R|^2 (the black total line traces the top of the stack). y-axis in percent.
        - error_composition_*:                  raw snapshots Q -> phase band is the lab-frame phase
                                                 error d_phi = phi_k^F - phi_k^R (location + shape).
        - error_fitted_snapshots_composition_*:  fitted snapshots Q_fitted = shift_func(Q, -c), each
                                                 model shifted back by its OWN predicted shift amount.
                                                 A shift leaves |a_k| unchanged and turns the phase
                                                 into the co-moving phi_k + k*c, so the phase band is
                                                 exactly the intrinsic (shape) phase error (Answer to
                                                 Question 2). The two figures differ only in the phase
                                                 band -- that difference is the bulk-translation part.
    Only |a_k| and angle(a_k) are needed (no wavenumber). One figure per ROM, since a stacked area
    cannot overlay multiple models legibly.
    """

    # Amplitude/phase composition of the FOM-vs-ROM error for each model in Q_seq, pointwise in
    # time: total = amplitude + (lab-frame) phase exactly at every t. Reference is Q_seq[0] (= FOM).
    def _composition_curves(Q_seq: List[Matrix]) -> Dict[int, Any]:
        a_ref = fft(Q_seq[0])                                 # (n_state, n_t) normalized Fourier coeffs
        A_ref, phi_ref = np.abs(a_ref), np.angle(a_ref)
        denom = np.sum(A_ref ** 2, axis=0)                    # (n_t,) = ||u_ref(t)||_2^2 by Parseval
        curves = {}
        for m in range(1, len(Q_seq)):
            a_m = fft(Q_seq[m])
            A_m, phi_m = np.abs(a_m), np.angle(a_m)
            amp = np.sum((A_ref - A_m) ** 2, axis=0) / denom
            phase = np.sum(2 * A_ref * A_m * (1 - np.cos(phi_ref - phi_m)), axis=0) / denom
            total = np.sum(np.abs(a_ref - a_m) ** 2, axis=0) / denom   # == amp + phase exactly
            curves[m] = (amp, phase, total)
        return curves

    # One stacked-area figure per ROM (the two bands sum to the black total line); y-axis in percent
    # with a shared upper limit ymax_by_m so the raw and fitted figures are directly comparable.
    def _plot_composition(curves: Dict[int, Any], fname_prefix: str, title_word: str, ymax_by_m: Dict[int, float]):
        for m, (amp, phase, total) in curves.items():
            plt.figure(figsize=(10, 6))
            plt.stackplot(time, amp, phase, labels=["amplitude", "phase"],
                          colors=["#1f77b4", "#d62728"], alpha=0.7)
            plt.plot(time, total, "k-", lw=1.2, label="total")
            plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            plt.xlabel(r"$t$", fontsize=20)
            plt.ylabel(r"relative error", fontsize=20)
            plt.title(rf"Trajectory {idx_traj:03d}, {model_name_list[m]}: {title_word}", fontsize=18)
            plt.xlim(time[0], time[-1])
            plt.ylim(0.0, ymax_by_m[m])
            plt.tick_params(labelsize=16)
            plt.legend(loc="upper left", fontsize=16)
            plt.tight_layout()
            plt.savefig(fig_path + f"{fname_prefix}_{model_abbrev_list[m]}_{idx_traj:03d}.png", dpi=300)
            plt.close()

    # (1) raw snapshots -> phase band = lab-frame phase error (location + shape);
    # (2) template-fitted snapshots Q_fitted = shift_func(Q, -c): a shift leaves |a_k| unchanged and
    #     turns the phase into the co-moving phi_k + k*c, so the phase band becomes the intrinsic
    #     (shape) phase error. The gap between (1) and (2) is the bulk-translation part.
    raw_curves = _composition_curves(Q_list)
    Q_fitted_list = [shift_func(Q_list[i], -shift_amount_list[i]) for i in range(len(Q_list))]
    fitted_curves = _composition_curves(Q_fitted_list)

    # Shared y upper limit per ROM = larger of the two figures' peak total (the stack top, since
    # total >= amplitude, phase), ceiled to a multiple of 5% (= 0.05 in the unnormalized fraction).
    # nan-aware: a diverged ROM's total is NaN past blow-up, so take the finite peak and fall back to
    # 5% when the whole curve is non-finite -- set_ylim must never see NaN.
    def _ceil5(*curves):
        vals = np.concatenate(curves)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or vals.max() <= 0:
            return 0.05
        return np.ceil(vals.max() / 0.05 - 1e-9) * 0.05
    ymax_by_m = {m: _ceil5(raw_curves[m][2], fitted_curves[m][2]) for m in raw_curves}

    _plot_composition(raw_curves, "error_composition", "error composition", ymax_by_m)
    _plot_composition(fitted_curves, "error_fitted_snapshots_composition", "fitted-snapshot error composition", ymax_by_m)

def plot_results(
    output_list: Dict[str, List[Matrix]],
    params,
    grid: Union[Grid1DUniformSpectral, Grid1DCubicSpline],
    idx_traj: int,
    fig_path: str,
    config_list: Dict[str, Any],
    scale_func: Callable[[Union[Vector, Matrix]], Union[Vector, Matrix]]):
    
    """
    model_name_list: e.g., ["FOM", "POD-Galerkin ROM", ...]
    model_abbrev_list: e.g., ["fom", "podgal", ...]
    
    Figures to be plotted:
    
    1. (x, t)-contourf plots of FOM vs ROM vs FOM-ROM difference in trajectory data
    2. spatial profile of solution at selected time instants for FOM vs all ROMs
    3. shift amount vs time, FOM vs all ROMs
    4. inv shift denom vs time, FOM vs all ROMs
    """
    
    x = grid.x
    tsave = params.tsave
    tplot = params.tplot
    tsave_upsample_factor = params.tsave_upsample_factor
    model_name_list = config_list["model_name_list"]
    model_abbrev_list = config_list["model_abbrev_list"]
    color_list = config_list["color_list"]
    linestyle_list = config_list["linestyle_list"]
    
    Q_scaled_list = output_list["Q_scaled_list"] # each entry: (n_state, n_t)
    Q_list = [scale_func(Q_scaled, action="inverse") for Q_scaled in Q_scaled_list] # each entry: (n_state, n_t)
    error_list = output_list["error_list"] # each entry: float scalar = relative RMSE
    c_list = output_list["c_list"] # each entry: (n_t,)
    inv_dcdt_denom_list = output_list["inv_dcdt_denom_list"] # each entry: (n_t,)

    # Upsample the time grid (cubic interpolation) ONLY for the shift-amount and error-decomposition
    # plots, so coarse-tsave (large nsave) curves render smoothly. The contourf / spatial-profile
    # plots below keep the raw tsave grid.
    if tsave_upsample_factor > 1:
        n_fine = (len(tsave) - 1) * tsave_upsample_factor + 1
        t_fine = np.linspace(tsave[0], tsave[-1], n_fine)
        Q_list_t = [_interp_time_valid(tsave, Q, t_fine) for Q in Q_list]
        c_list_t = [_interp_time_valid(tsave, c, t_fine) for c in c_list]
        inv_dcdt_denom_list_t = [_interp_time_valid(tsave, d, t_fine) for d in inv_dcdt_denom_list]
    else:
        t_fine = tsave
        Q_list_t, c_list_t, inv_dcdt_denom_list_t = Q_list, c_list, inv_dcdt_denom_list

    # 1. (x, t)-contourf plots of FOM vs ROM vs FOM-ROM difference in trajectory data
    plot_traj_contourf_xt(
        Q_fom = Q_list[0],
        Q_rom = Q_list[-1],
        error_rom = error_list[-1],
        x = x,
        t = tsave,
        model_name_rom = model_name_list[-1],
        model_abbrev_rom = model_abbrev_list[-1],
        fig_path = fig_path,
        idx_traj = idx_traj
    )
    
    # 2. spatial profile of solution at selected time instants for FOM vs all ROMs
    idx_tplot = [int(np.argmin(np.abs(tsave - tp))) for tp in tplot]
    Q_tplot_list = [Q[:, idx_tplot] for Q in Q_list]

    plot_traj_x(
        Q_list = Q_tplot_list,
        error_list = error_list,
        x = x,
        t = tplot,
        model_name_list = model_name_list,
        color_list = color_list,
        linestyle_list = linestyle_list,
        fig_path = fig_path,
        idx_traj = idx_traj
    )
    
    # 5. shift amount and inverse of shift speed denominator vs time of FOM and all ROMs 
    plot_shift_t(
        shift_amount_list = c_list_t,
        inv_shift_speed_denom_list = inv_dcdt_denom_list_t,
        time = t_fine,
        model_name_list = model_name_list,
        color_list = color_list,
        linestyle_list = linestyle_list,
        fig_path = fig_path,
        idx_traj = idx_traj
    )
    
    # FFT-based phase/amplitude decomposition relies on Parseval (||u||^2 = sum_k |a_k|^2),
    # which is exact only on the uniform spectral grid (grid.kx defines the wavenumbers).
    if isinstance(grid, Grid1DUniformSpectral):
        plot_error_phase_amplitude_t(
            Q_list = Q_list_t,
            shift_amount_list = c_list_t,
            shift_func = grid.shift_x,
            time = t_fine,
            model_name_list = model_name_list,
            model_abbrev_list = model_abbrev_list,
            fig_path = fig_path,
            idx_traj = idx_traj
        )

def plot_cumulative_rRMSE_band_t(
    model_name_list: List[str],
    model_abbrev_list: List[str],
    color_list: List[str],
    num_traj: int,
    tsave: Vector,
    fname_traj_scaled: str,
    fom_abbrev: str,
    fig_path: str,
    dataset_label: str = "training",
):
    """Trajectory-wise band of cumulative relative RMSE over time, one band per ROM in
    model_abbrev_list. For trajectory j and model m, in scaled coordinates
        cum_rRMSE(t) = sqrt( sum_{s<=t} ||q_rom - q_fom||_2^2(s) / sum_{s<=t} ||q_fom||_2^2(s) ),
    which equals sample_and_compare's scalar relative RMSE at t = T. The shaded band spans
    [min, max] over the num_traj trajectories at each t; the solid line is their mean. If ANY
    trajectory has blown up (NaN) by time t, the whole band/mean is NaN from that t on -- the
    diverged trajectory is NOT dropped, so the band ends at the earliest blow-up (count in legend)."""
    def _cum_rrmse(Q_rom: Matrix, Q_fom: Matrix) -> Vector:
        err2 = np.sum((Q_rom - Q_fom) ** 2, axis=0)        # column-wise ||.||_2^2 (scaled coords)
        ref2 = np.sum(Q_fom ** 2, axis=0)
        return np.sqrt(np.cumsum(err2) / np.cumsum(ref2))

    plt.figure(figsize=(10, 6))
    legend_handles = []
    for m, (name, abbrev) in enumerate(zip(model_name_list, model_abbrev_list)):
        curves = np.full((num_traj, len(tsave)), np.nan)
        for j in range(num_traj):
            Q_rom = np.load(fname_traj_scaled % (abbrev, j))
            Q_fom = np.load(fname_traj_scaled % (fom_abbrev, j))
            curves[j] = _cum_rrmse(Q_rom, Q_fom)
        n_div = int(np.sum(~np.isfinite(curves[:, -1])))
        # If ANY trajectory turns NaN at time t (blow-up), the whole band is NaN from t onward:
        # do NOT drop the diverged trajectory / keep averaging survivors. Plain min/max/mean
        # propagate NaN, so the band stops at the earliest blow-up time across the set.
        lo = np.min(curves, axis=0)
        hi = np.max(curves, axis=0)
        mean = np.mean(curves, axis=0)
        # bands carry a solid pre-blended color (EPS renders alpha as opaque) and sit at zorder=1,
        # below the mean lines at zorder=3 -- otherwise a later band occludes an earlier mean line
        band = plt.fill_between(tsave, lo, hi, color=_on_white(color_list[m], 0.2), linewidth=0, zorder=1,
                                label=f"{name} ([min, max]; {n_div}/{num_traj} diverged)" if n_div else f"{name} ([min, max])")
        # thin min/max boundary lines (zorder=2) so each band's extent stays readable even where a
        # later, opaque (EPS-safe) band overlaps and hides its fill
        plt.plot(tsave, lo, color=color_list[m], lw=0.8, zorder=2)
        plt.plot(tsave, hi, color=color_list[m], lw=0.8, zorder=2)
        line, = plt.plot(tsave, mean, color=color_list[m], lw=2, zorder=3,
                         label=f"{name} (mean; {n_div}/{num_traj} diverged)" if n_div else f"{name} (mean)")
        legend_handles += [line, band]
    plt.yscale("log")                     # log y: early small-error ramp and the ~1 (100%) tail both legible
    plt.ylim(1e-4, 1e1)                   # fixed decade range (1e-4..1e1) so every band figure is comparable
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:g}%"))  # 0.1% 1% 10% 100% 1000%
    # plt.xlabel(r"$t$", fontsize=20)
    # plt.ylabel("cumulative relative RMSE", fontsize=20)
    # plt.title(f"Trajectory-wise cumulative rRMSE band over {num_traj} {dataset_label} trajectories", fontsize=18)
    plt.xlim(tsave[0], tsave[-1])
    plt.tick_params(labelsize=16)
    # legend per model: solid line = mean, matching shaded band = [min, max] over trajectories
    # framealpha=1: the default 0.8 is a transparent artist, which the EPS backend cannot render
    plt.legend(handles=legend_handles, loc="upper left", fontsize=12, framealpha=1.0)
    plt.tight_layout()
    plt.savefig(fig_path + f"cumulative_rrmse_band_{dataset_label}.png", dpi=300)
    plt.savefig(fig_path + f"cumulative_rrmse_band_{dataset_label}.eps", dpi=300)
    plt.close()