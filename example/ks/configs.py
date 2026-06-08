"""
config.py --- configuration for the Kuramoto-Sivashinsky equation on the 1D periodic grid with different shift/interpolation/spatial derivative operators, used in the test files for the grid classes and the model class.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BASE_PATH = os.environ.get(
    "NISRH_OUTPUT_DIR",
    # os.path.join(_ROOT, "output", "ks_base_solution"),
    os.path.join(_ROOT, "output", "ks_perturbed_solutions"),
).rstrip("/") + "/"

@dataclass
class PathConfig:
    """File-system paths for data, figures, and trajectories."""
    base_path: str = _BASE_PATH

    def __post_init__(self):
        b = self.base_path
        self.data_path               = b + "data/"
        
        self.fig_path_fom       = b + "figures/fom/"
        self.fig_path_fom_csp = b + "figures/fom_cubic_spline/"
        self.fig_path_srpodgal_spec         = b + "figures/srpodgal_spectral/"
        self.fig_path_sropinf_spec        = b + "figures/sropinf_spectral/"
        self.fig_path_sropinf_spec_reproj = b + "figures/sropinf_spectral_reproj/"
        self.fig_path_sropinf_spec_reg = b + "figures/sropinf_spectral_regularized/"
        self.fig_path_srpodgal_csp = b + "figures/srpodgal_cubic_spline/"
        self.fig_path_sropinf_csp = b + "figures/sropinf_cubic_spline/"
        self.fig_path_sropinf_csp_reproj = b + "figures/sropinf_cubic_spline_reproj/"
        self.fig_path_sropinf_csp_reg = b + "figures/sropinf_cubic_spline_regularized/"
        self.fig_path_fom_testing      = b + "figures/fom_testing/"
        self.fig_path_srpodgal_spec_testing         = b + "figures/srpodgal_spectral_testing/"
        self.fig_path_sropinf_spec_reg_testing = b + "figures/sropinf_spectral_regularized_testing/"
        self.fig_path_srpodgal_spec_testing_denom_reg = b + "figures/srpodgal_spectral_testing_denominator_regularized/"

        self.traj_path_fom            = b + "trajectories/fom/"
        self.traj_path_fom_csp = b + "trajectories/fom_cubic_spline/"
        self.traj_path_srpodgal_spec         = b + "trajectories/srpodgal_spectral/"
        self.traj_path_sropinf_spec        = b + "trajectories/sropinf_spectral/"
        self.traj_path_sropinf_spec_reproj = b + "trajectories/sropinf_spectral_reproj/"
        self.traj_path_sropinf_spec_reg = b + "trajectories/sropinf_spectral_regularized/"
        self.traj_path_srpodgal_csp = b + "trajectories/srpodgal_cubic_spline/"
        self.traj_path_sropinf_csp = b + "trajectories/sropinf_cubic_spline/"
        self.traj_path_sropinf_csp_reproj = b + "trajectories/sropinf_cubic_spline_reproj/"
        self.traj_path_sropinf_csp_reg = b + "trajectories/sropinf_cubic_spline_regularized/"
        self.traj_path_fom_testing      = b + "trajectories/fom_testing/"
        self.traj_path_srpodgal_spec_testing         = b + "trajectories/srpodgal_spectral_testing/"
        self.traj_path_sropinf_spec_reg_testing = b + "trajectories/sropinf_spectral_regularized_testing/"
        self.traj_path_srpodgal_spec_testing_denom_reg = b + "trajectories/srpodgal_spectral_testing_denominator_regularized/"
        
@dataclass
class FOMConfig:
    """Physical and discretization parameters for the full-order model."""
    Lx: float = 2 * np.pi
    nx: int   = 256
    nu: float = 4/87
    T:  float = 10.0
    dt: float = 0.001
    nsave: int = 100 # for snapshot saving and pod
    
    def __post_init__(self):
        self.time  = self.dt * np.linspace(0, int(self.T / self.dt), int(self.T / self.dt) + 1)
        self.tsave = self.time[::self.nsave] # the timepoints where we save snapshots for training and for plotting x-t contours;
        self.tsave_upsample_factor = 10 # the factor by which we upsample 'tsave' for plotting x-axis-time curves of time-evolving quantities
        self.tplot = self.tsave[np.isclose(self.tsave, np.round(self.tsave))] # the subset of saved timepoints that are integers for plotting x-axis-space spatial curves of spatially-varying functions
        
@dataclass
class DataConfig:
    """Parameters for data generation, ROM benchmarking, and phase alignment."""
    type_traj_training: str = "perturbed_solutions" # "base_solution" or "perturbed_solutions"
    # type_traj_training: str = "base_solution" # "base_solution" or "perturbed_solutions"
    def __post_init__(self):
        if self.type_traj_training == "base_solution":
            self.num_traj_training = 1
            self.shift_speed_denom_threshold = 0.0 # always 0 when trying to reconstruct the benchmark trajectory
        elif self.type_traj_training == "perturbed_solutions":
            self.random_seed_training = 2000
            self.random_seed_testing = 523
            self.num_traj_training = 20
            self.num_traj_testing = 11 # 11 is good for plotting the box plots
            self.shift_speed_denom_threshold = 1e-1 * 0.0 # 1e-1 or 0, these are 2 cases for comparison
            self.rom_state_norm_threshold = 1e2
            self.rom_solve_wall_time_limit = 10.0
    
@dataclass
class TrainingConfig: 
    """We are going to determine the dimension of this preliminary subspace via the diagram of singular value decay"""
    pod_energy_threshold: float = 1.0 - 1e-6
    # pod_energy_threshold: float = 1.0 - 1e-4
    training_perturbation_to_benchmark_ratio: float = 0.1
    testing_perturbation_to_benchmark_ratio: float = 0.20 # (12, 14, 16, 18, 20)
    opinf_CV_random_seed: int = 42
    
    # for base solution case, the optimal regularizers from grid search is:
    # (lambda_poly, lambda_dcdt_numer) = (1.56e-1, 6.90e-6)
    
    # opinf_regularizer_rhs_poly: List[float] = field(default_factory=lambda: list(np.linspace(120e-4, 140e-4, 21, endpoint=True)))
    # opinf_regularizer_dcdt_numer: List[float] = field(default_factory=lambda: list(np.linspace(110e-3, 130e-3, 21, endpoint=True)))
    # opinf_penalty_weight_rhs_poly: float = 1.56e-1
    # opinf_penalty_weight_dcdt_numer: float = 6.90e-6
    opinf_penalty_weight_rhs_poly: float = 1.30e-2
    opinf_penalty_weight_dcdt_numer: float = 1.24e-1
    
    r: int = 1000 # tentative dimension before running POD
    poly_comp: List[int] = field(default_factory=lambda: [1, 2])
    
# ---------------------------------------------------------------------------
# Top-level config — flat access: params.Re, params.kouter, params.data_path
# ---------------------------------------------------------------------------

@dataclass
class SimConfigs:
    paths:    PathConfig     = field(default_factory=PathConfig)
    fom:      FOMConfig      = field(default_factory=FOMConfig)
    data:     DataConfig     = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __getattr__(self, name: str):
        # Called only when normal lookup (own __dict__ + class) fails.
        for sub in (self.paths, self.fom, self.data, self.training):
            try:
                return getattr(sub, name)
            except AttributeError:
                continue
        raise AttributeError(f"'SimConfigs' has no attribute '{name}'")

    def __post_init__(self):
        # Create output directories
        for attr in vars(self.paths).values():
            if isinstance(attr, str):
                os.makedirs(attr, exist_ok=True)
                                
        d = self.data_path
        b = self.base_path
        
        self.fname_time      = d + "time.npy"
        self.fname_traj_init = d + "traj_init_%03d.npy"
        self.fname_traj_init_scaled = d + "traj_init_scaled_%03d.npy"
        self.fname_traj_init_testing = d + "traj_init_testing_%03d.npy"
        self.fname_traj_init_testing_scaled = d + "traj_init_testing_scaled_%03d.npy"
        self.fname_traj_init_fitted = d + "traj_init_fitted_%03d.npy"
        self.fname_traj_init_fitted_scaled = d + "traj_init_fitted_scaled_%03d.npy"
        self.fname_traj_init_fitted_scaled_csp = d + "traj_init_fitted_scaled_cubic_spline_%03d.npy"
        self.fname_traj_init_testing_fitted = d + "traj_init_testing_fitted_%03d.npy"
        self.fname_traj_init_testing_fitted_scaled = d + "traj_init_testing_fitted_scaled_%03d.npy"
        
        self.fname_template = d + "template.npy"
        self.fname_template_perp = d + "template_perp.npy"
        self.fname_template_dx = d + "template_dx.npy"
        self.fname_template_dxx = d + "template_dxx.npy"
        self.fname_template_scaled = d + "template_scaled.npy"
        self.fname_template_perp_scaled = d + "template_perp_scaled.npy"
        self.fname_template_dx_scaled = d + "template_dx_scaled.npy"
        self.fname_template_dxx_scaled = d + "template_dxx_scaled.npy"
        
        self.fname_traj = b + "trajectories/%s/traj_%03d.npy"
        self.fname_traj_scaled = b + "trajectories/%s/traj_scaled_%03d.npy"
        self.fname_traj_fitted = b + "trajectories/%s/traj_fitted_%03d.npy"
        self.fname_traj_fitted_scaled = b + "trajectories/%s/traj_fitted_scaled_%03d.npy"
        self.fname_rhs_fitted_scaled = b + "trajectories/%s/rhs_fitted_scaled_%03d.npy"
        
        self.fname_error = b + "trajectories/%s/error_%03d.npy"
        self.fname_shift_amount = b + "trajectories/%s/shift_amount_%03d.npy"
        self.fname_inv_shift_speed_denom = b + "trajectories/%s/inv_shift_speed_denom_%03d.npy"
        
def load_configs():
    return SimConfigs()
